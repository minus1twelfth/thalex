import asyncio
import json
import logging

import websockets
from aiohttp import web
from aiohttp.web_runner import AppRunner, TCPSite

from common import InstrumentType, neighbours
from quoter import Quoter

VOL_OFFSET_D25 = 'vol offset d25'
VOL_OFFSET_D50 = 'vol offset d50'
VOL_OFFSET_D75 = 'vol offset d75'
VOL_OFFSETS = [VOL_OFFSET_D25, VOL_OFFSET_D50, VOL_OFFSET_D75]


async def handle_http(request):
    match request.path:
        case '/':
            return web.FileResponse('index.html')
        case '/script.js':
            return web.FileResponse('script.js')
    return web.HTTPNotFound()


async def try_read_ws_message(websocket):
    try:
        return json.loads(await asyncio.wait_for(websocket.recv(), timeout=1))
    except asyncio.TimeoutError:
        return None


def format_num(num, precision):
    if num is None:
        return ' '
    return f'{num:.{precision}f}'


class Gui:
    def __init__(self, quoter: Quoter, app: web.Application, runner: AppRunner, site: TCPSite):
        self._quoter = quoter
        self._app = app
        self._runner = runner
        self._site = site
        self._should_send_config = True

    @staticmethod
    async def create(quoter: Quoter):
        app = web.Application()
        app.router.add_get('/', handle_http)
        app.router.add_get('/script.js', handle_http)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8500)
        await site.start()
        logging.info("http server started")
        return Gui(quoter, app, runner, site)

    async def stop(self):
        await self._runner.cleanup()

    async def run_websocket_server(self):
        await websockets.serve(self._websocket_handler, "localhost", 8501)

    async def _send_config(self, websocket):
        cfg = self._quoter.cfg
        await websocket.send(json.dumps({'min_delta': cfg.min_delta}))
        await websocket.send(json.dumps({'max_delta': cfg.max_delta}))
        await websocket.send(json.dumps({'width_bid_call': cfg.width_bid_call}))
        await websocket.send(json.dumps({'width_ask_call': cfg.width_ask_call}))
        await websocket.send(json.dumps({'width_bid_put': cfg.width_bid_put}))
        await websocket.send(json.dumps({'width_ask_put': cfg.width_ask_put}))
        await websocket.send(json.dumps({'mmp_size': cfg.mmp_size}))
        await websocket.send(json.dumps({'quote_size': cfg.quote_size}))
        await websocket.send(json.dumps({'label': cfg.label}))
        await websocket.send(json.dumps({'expiry_data': self._get_expiry_table_data()}))

    def _get_expiry_table_data(self):
        cfg = self._quoter.cfg
        result = []
        expiries = list(self._quoter.instruments.keys())
        expiries.sort()
        for exp in expiries:
            exp = str(exp)
            enabled = exp in cfg.enabled_expiries
            exp_object = {'expiry': exp, 'enabled': enabled}
            vol_offsets = [100 * v for v in cfg.vol_offsets.get(exp, [0, 0, 0])]
            for key, value in zip(VOL_OFFSETS, vol_offsets):
                exp_object[key] = value
            result.append(exp_object)
        return result

    async def _websocket_handler(self, websocket):
        await self._send_config(websocket)
        try:
            while True:
                if self._should_send_config:
                    await self._send_config(websocket)
                    self._should_send_config = False
                await websocket.send(json.dumps({'table_data': self._get_instrument_table_data()}))
                await websocket.send(json.dumps({'armed': self._quoter.armed}))
                await websocket.send(json.dumps({'open_orders': self._quoter.count_open_orders()}))
                if message := await try_read_ws_message(websocket):
                    logging.info(f'{message=}')
                    cfg = self._quoter.cfg
                    match message.get('type'):
                        case 'armed_checkbox':
                            self._quoter.armed = message['status']
                        case 'min_delta':
                            cfg.min_delta = float(message['value'])
                            self.config_updated()
                        case 'max_delta':
                            cfg.max_delta = float(message['value'])
                            self.config_updated()
                        case 'width_bid_call':
                            cfg.width_bid_call = float(message['value'])
                            self.config_updated()
                        case 'width_ask_call':
                            cfg.width_ask_call = float(message['value'])
                            self.config_updated()
                        case 'width_bid_put':
                            cfg.width_bid_put = float(message['value'])
                            self.config_updated()
                        case 'width_ask_put':
                            cfg.width_ask_put = float(message['value'])
                            self.config_updated()
                        case 'mmp_size':
                            cfg.mmp_size = float(message['value'])
                            self.config_updated()
                            self._quoter.armed = False
                        case 'quote_size':
                            cfg.quote_size = float(message['value'])
                            self.config_updated()
                        case 'label':
                            cfg.label = message['value']
                            self.config_updated()
                            self._quoter.armed = False
                        case 'expiry_enabled':
                            enabled = message['value']
                            cfg.enable_expiry(expiry=message['exp'], enabled=enabled)
                            self.config_updated()
                            if enabled:
                                self._quoter.armed = False  # prevent blind quoting
                        case 'vol_offset':
                            value = float(message['value']) / 100.0
                            idx = VOL_OFFSETS.index(message['range'])
                            exp = message['exp']
                            cfg.set_vol_offset(expiry=exp, idx=idx, value=value)
                            self.config_updated()

        except websockets.exceptions.ConnectionClosed:
            logging.info('Client connection closed')

    def _get_instrument_table_data(self):
        result = []
        instruments = self._quoter.instruments
        cfg = self._quoter.cfg
        quotes = self._quoter.quotes
        portfolio = self._quoter.portfolio
        index = self._quoter.index
        expiries = list(instruments.keys())
        expiries.sort()

        for exp in expiries:
            if not cfg.expiry_is_enabled(str(exp)):
                continue
            exp_instr = instruments[exp]
            strikes = list(exp_instr.keys())
            strikes.sort()
            result.append({'iv off': ' ', 'C pos': ' ', 'C vol bid': ' ', 'C bid': ' ', 'C bbid': ' ', 'C bask': ' ',
                           'C ask': ' ', 'C vol ask': ' ', 'C delta': ' ', 'strike': exp.name, 'P delta': ' ',
                           'P vol bid': ' ', 'P bid': ' ', 'P bbid': ' ', 'P bask': ' ', 'P ask': ' ', 'P vol ask': ' ',
                           'P pos': ' ', })
            atm_lo, atm_hi = neighbours(strikes, index)
            for strike in strikes:
                row = {}
                call = exp_instr[strike][InstrumentType.CALL]
                quote = quotes.get(call)
                if quote and quote.iv_off > 0:
                    iv_c = '.green'
                elif quote and quote.iv_off < 0:
                    iv_c = '.red'
                else:
                    iv_c = ''
                row[f'iv off{iv_c}'] = f'{format_num(quote and quote.iv_off * 100, 1)}%'
                row['C pos'] = format_num(portfolio.get(call), 1)
                row['C vol bid'] = f'{format_num(quote and quote.vols[0] and quote.vols[0] * 100, 1)}%'
                if quote and quote.book[0]:
                    row['C bid.green'] = repr(quote.book[0])
                elif quote and quote.theo and quote.theo.b and quote.theo.b.a:
                    row['C bid'] = repr(quote.theo.b)
                else:
                    row['C bid'] = ' '
                row['C bbid'] = quote and quote.tob[0] or ' '
                row['C bask'] = quote and quote.tob[1] or ' '
                if quote and quote.book[1]:
                    row['C ask.red'] = repr(quote.book[1])
                elif quote and quote.theo and quote.theo.a and quote.theo.a.a:
                    row['C ask'] = repr(quote.theo.a)
                else:
                    row['C ask'] = ' '
                row['C vol ask'] = f'{format_num(quote and quote.vols[1] and quote.vols[1] * 100, 1)}%'
                row['C delta'] = format_num(quote and quote.delta, 2)
                if strike in [atm_lo, atm_hi]:
                    row['strike.blue'] = strike
                else:
                    row['strike'] = strike
                put = exp_instr[strike][InstrumentType.PUT]
                quote = quotes.get(put)
                row['P delta'] = format_num(quote and quote.delta, 2)
                row['P vol bid'] = f'{format_num(quote and quote.vols[0] and quote.vols[0] * 100, 1)}%'
                if quote and quote.book[0]:
                    row['P bid.green'] = repr(quote.book[0])
                elif quote and quote.theo and quote.theo.b and quote.theo.b.a:
                    row['P bid'] = repr(quote.theo.b)
                else:
                    row['P bid'] = ' '
                row['P bbid'] = quote and quote.tob[0] or ' '
                row['P bask'] = quote and quote.tob[1] or ' '
                if quote and quote.book[1]:
                    row['P ask.red'] = repr(quote.book[1])
                elif quote and quote.theo and quote.theo.a and quote.theo.a.a:
                    row['P ask'] = repr(quote.theo.a)
                else:
                    row['P ask'] = ' '
                row['P vol ask'] = f'{format_num(quote and quote.vols[1] and quote.vols[1] * 100, 1)}%'
                row['P pos'] = format_num(portfolio.get(put), 1)
                result.append(row)
            result.append({})
        return result

    def config_updated(self):
        self._quoter.cfg.persist()
        self._should_send_config = True
