import asyncio
import datetime
import json
import logging
import socket
import math
import time
from aiohttp import web

import websockets
from thalex import SideQuote

import keys
import thalex as th

from common import *
from deribit import Deribit
from settings import default_settings, Settings

MAX_DTE = 7
BATCH = 100  # Mass Quote batch size
DISCO_SECS = 10  # cancel on disconnect seconds
MIN_POS = -0.1
MAX_POS = 0.1

AMEND_THRESHOLD = 15  # $USD

DERIBIT_URL = "wss://www.deribit.com/ws/api/v2"
NETWORK = th.Network.DEV
UNDERLYING = "BTC"
PRODUCT = "OBTCUSD"
assert UNDERLYING in PRODUCT


async def handle_http(request):
    return web.FileResponse('index.html')


def round_to_tick(value):
    tick = 5
    return tick * round(value / tick)


def neighbours(chain, tgt_k) -> [float, float]:
    n_down = None
    n_up = None
    for k in chain:
        if k == tgt_k:
            return tgt_k, tgt_k
        elif k < tgt_k:
            n_down = k
        else:
            n_up = k
            break
    return n_down, n_up


def call_discount(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        norm_d1 = 0.5 + 0.5 * math.erf(d1 / math.sqrt(2))
        norm_d1_vol = 0.5 + 0.5 * math.erf((d1 - voltime) / math.sqrt(2))
        return fwd * norm_d1 - k * norm_d1_vol
    elif fwd > k:
        return fwd - k
    else:
        return 0.0


def put_discount(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        norm_d1 = 0.5 + 0.5 * math.erf(-d1 / math.sqrt(2))
        norm_d1_vol = 0.5 + 0.5 * math.erf((voltime - d1) / math.sqrt(2))
        return k * norm_d1_vol - fwd * norm_d1
    elif fwd > k:
        return k - fwd
    else:
        return 0.0


def iv_offset_for_delta(vol_offsets: list[float], delta: float):
    assert delta > 0
    if delta <= 0.5:
        return vol_offsets[0] + (vol_offsets[1] - vol_offsets[0]) * 4 * max(delta - 0.25, 0)
    else:
        return vol_offsets[1] + (vol_offsets[2] - vol_offsets[1]) * 4 * min(delta - 0.5, 0.25)

def quote_needs_update(book: Optional[SideQuote], theo: SideQuote):
    assert theo is not None
    if book is None:
        return theo.a > 0 # insert
    if theo.a > 0:
        return not theo.p - AMEND_THRESHOLD < book.p < theo.p + AMEND_THRESHOLD # amend
    else:
        return True # delete

class QuoteMeta:
    def __init__(self, instrument: Instrument):
        self.theo = th.Quote(instrument.name, None, None)
        self.delta: Optional[float] = None
        self.fwd: Optional[float] = None
        self.book: list[Optional[th.SideQuote]] = [None, None]  # bid, ask
        self.vols: list[Optional[float]] = [None, None]  # bid ask
        self.in_flight: bool = False
        self.queued: bool = False
        self.instrument: Instrument = instrument

    def should_send(self) -> bool:
        if self.in_flight or self.queued:
            return False
        return quote_needs_update(self.book[0], self.theo.b) or quote_needs_update(self.book[1], self.theo.a)

    def update_theo(self, index: float, now: float, pp: float, cfg: Settings):
        logging.debug(f"{index=} {self.instrument.name} {self.delta=} {self.fwd=} {self.vols=}")
        if self.delta is None or self.fwd is None or not cfg.min_delta < abs(self.delta) < cfg.max_delta:
            self.theo.b = th.SideQuote(0, 0)
            self.theo.a = th.SideQuote(0, 0)
            return
        tte = (self.instrument.exp - now) / (3600 * 24 * 365.25)
        vol_offsets = cfg.vol_offsets.get(self.instrument.exp_str, [0, 0, 0])
        if self.instrument.type == InstrumentType.CALL:
            if pp < MAX_POS:
                iv = self.vols[0] + iv_offset_for_delta(vol_offsets, self.delta)
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, iv, tte)) - cfg.width_bid_call
                self.theo.b = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                iv = self.vols[1] + iv_offset_for_delta(vol_offsets, self.delta)
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, iv, tte)) + cfg.width_ask_call
                self.theo.a = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        elif self.instrument.type == InstrumentType.PUT:
            if pp < MAX_POS:
                iv = self.vols[0] + iv_offset_for_delta(vol_offsets, 1 - self.delta)
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, iv, tte)) - cfg.width_bid_put
                self.theo.b = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                iv = self.vols[1] + iv_offset_for_delta(vol_offsets, 1 - self.delta)
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, iv, tte)) + cfg.width_ask_put
                self.theo.a = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        logging.debug(f"{index=} {self.fwd=} {self.vols=} {self.instrument.name} theo: {self.theo}")


class Quoter:
    def __init__(self, iv_store: IvStore, thalex: th.Thalex, cfg: Settings):
        self.thalex = thalex
        self.cfg = cfg
        self._iv_store: IvStore = iv_store
        self._instruments: dict[str, dict[float, dict[InstrumentType, str]]] = {}
        self._quotes: dict[str, QuoteMeta] = {}
        self._send_queue: list[QuoteMeta] = []
        self._index: Optional[float] = None
        self._armed: bool = False
        self.portfolio: dict[str, float] = {}

    async def read_task(self):
        await self.thalex.connect()
        await self.thalex.login(
            keys.key_ids[NETWORK],
            keys.private_keys[NETWORK],
            id=CID_LOGIN,
        )
        await self.thalex.set_cancel_on_disconnect(DISCO_SECS, CID_CANCEL_DISCO)
        await self.thalex.instruments(CID_INSTRUMENTS)
        await self.thalex.public_subscribe([f"price_index.{UNDERLYING}USD"], CID_SUBSCRIBE)
        await self.thalex.private_subscribe(["session.orders", "account.portfolio"], id=CID_SUBSCRIBE)
        while True:
            msg = json.loads(await self.thalex.receive())
            cid = msg.get("id", CID_IGNORE)
            if cid == CID_INSTRUMENTS:
                subs = self.proc_instruments(msg["result"])
                await self.thalex.public_subscribe(subs, CID_SUBSCRIBE)
            elif cid == CID_SUBSCRIBE:
                logging.debug(f"sub result: {msg}")
            elif cid == CID_QUOTE:
                if "error" in msg:
                    logging.warning(f"Quote error: {msg}")
                logging.debug(msg)
            elif "notification" in msg:
                self.proc_notification(msg["channel_name"], msg["notification"])
            else:
                logging.warning(f"Unhandled msg: {msg}")

    async def update_task(self):
        while True:
            await asyncio.sleep(1)
            self.update_vols()

    def proc_notification(self, ch, n):
        if ch.startswith("price_index"):
            self._index = n["price"]
        elif ch == "session.orders":
            mmp_hit = False
            for o in n:
                if o.get("delete_reason", "") == "mm_protection" and not mmp_hit:
                    logging.warning("MMP is hit. Disarming.")
                    self._armed = False
                    mmp_hit = True
                q = self._quotes[o["instrument_name"]]
                side = 0 if o["direction"] == "buy" else 1
                if o["status"] in ["open", "partially_filled"]:
                    q.book[side] = th.SideQuote(o["price"], o["remaining_amount"])
                else:
                    q.book[side] = None
                q.in_flight = False
        elif ch.startswith("ticker"):
            iname = ch.split(".")[1]
            q = self._quotes[iname]
            q.delta = n["delta"]
            q.fwd = n["forward"]
        elif ch == "account.portfolio":
            for pp in n:
                self.portfolio[pp["instrument_name"]] = pp["position"]

    async def send_task(self):
        is_quoting = False
        while True:
            await asyncio.sleep(0.25)
            if not self._armed:
                if is_quoting:
                    await self.thalex.cancel_mass_quote()
                    is_quoting = False
                continue
            elif not is_quoting:
                await self.thalex.set_mm_protection(PRODUCT, self.cfg.mmp_size, self.cfg.mmp_size, id=CID_MMP)
                is_quoting = True
            queue = self._send_queue
            self._send_queue = []
            for i in range(0, len(queue), BATCH):
                batch = queue[i:i + BATCH]
                quotes = [el.theo for el in batch]
                for el in batch:
                    el.in_flight = True
                    el.queued = False
                logging.info(f'Sending {len(quotes)} quotes')
                await self.thalex.mass_quote(quotes, post_only=True, id=CID_QUOTE, label=self.cfg.label)

    def proc_instruments(self, instruments):
        now = datetime.datetime.now(datetime.UTC).timestamp()
        subs = []
        for i in instruments:
            expiry = i.get("expiration_timestamp", now + MAX_DTE + 50000)
            if i["product"] == PRODUCT and i["type"] == "option" and expiry < now + MAX_DTE * 24 * 3600:
                exp_str = i["expiry_date"]
                i = Instrument(
                    name=i["instrument_name"],
                    expiry=expiry,
                    exp_str=exp_str,
                    itype=InstrumentType(i["option_type"]),
                    k=i["strike_price"]
                )
                if exp_str not in self._instruments:
                    self._instruments[exp_str] = {}
                if i.k not in self._instruments[exp_str]:
                    self._instruments[exp_str][i.k] = {}
                self._instruments[exp_str][i.k][i.type] = i.name
                self._quotes[i.name] = QuoteMeta(i)
                subs.append(f"ticker.{i.name}.500ms")
        return subs

    def update_vols(self):
        now = datetime.datetime.now(datetime.UTC).timestamp()
        nothave = []
        for exp_str, chain in self._instruments.items():
            iv_chain = self._iv_store.get(exp_str)
            if iv_chain is None:
                if exp_str not in nothave:
                    nothave.append(exp_str)
            else:
                d_chain = sorted(iv_chain.keys())
                for k, putcall in chain.items():
                    k_down, k_up = neighbours(d_chain, k)
                    if k_down is None or k_up is None:
                        continue
                    for pc, iname in putcall.items():
                        nup = iv_chain[k_up]
                        ndown = iv_chain[k_down]
                        if pc not in ndown or pc not in nup:
                            continue
                        k_diff = k_up - k_down
                        q = self._quotes[iname]
                        if k_diff == 0:
                            bid = ndown[pc].bid_iv
                            ask = ndown[pc].ask_iv
                        else:
                            bid_iv_down = ndown[pc].bid_iv
                            bid_iv_up = nup[pc].bid_iv
                            ask_iv_down = ndown[pc].ask_iv
                            ask_iv_up = nup[pc].ask_iv
                            bid = (bid_iv_down * (k_up-k) + bid_iv_up * (k-k_down)) / k_diff
                            ask = (ask_iv_down * (k_up-k) + ask_iv_up * (k-k_down)) / k_diff
                        q.vols[0] = bid
                        q.vols[1] = ask
                        if self._index is None:
                            continue
                        pp = self.portfolio.get(q.instrument.name, 0)
                        q.update_theo(self._index, now, pp, self.cfg)
                        if q.should_send() and self._armed:
                            q.queued = True
                            self._send_queue.append(q)

        logging.debug(f"Deribit doesn't have expiries: {nothave}")

    def get_instrument_table_data(self):
        result = []
        expiries = list(self._instruments.keys())
        expiries.sort()

        for exp in expiries:
            exp_instr = self._instruments[exp]
            strikes = list(exp_instr.keys())
            strikes.sort()
            for strike in strikes:
                row = {}
                call = exp_instr[strike][InstrumentType.CALL]
                quote = self._quotes.get(call)
                row['C vol bid'] = f'{format_num(quote and quote.vols[0] * 100, 1)}%'
                if quote and quote.book[0]:
                    row['C bid.green'] = repr(quote.book[0])
                elif quote and quote.theo and quote.theo.b.a:
                    row['C bid'] = repr(quote.theo.b)
                else:
                    row['C bid'] = '-'
                if quote and quote.book[1]:
                    row['C ask.red'] = repr(quote.book[1])
                elif quote and quote.theo and quote.theo.a.a:
                    row['C ask'] = repr(quote.theo.a)
                else:
                    row['C ask'] = '-'
                row['C vol ask'] = f'{format_num(quote and quote.vols[1] * 100, 1)}%'
                row['C delta'] = format_num(quote and quote.delta, 2)
                row['instrument'] = call[:-2]
                put = exp_instr[strike][InstrumentType.PUT]
                quote = self._quotes.get(put)
                row['P delta'] = format_num(quote and quote.delta, 2)
                row['P vol bid'] = f'{format_num(quote and quote.vols[0] * 100, 1)}%'
                if quote and quote.book[0]:
                    row['P bid.green'] = repr(quote.book[0])
                elif quote and quote.theo and quote.theo.b.a:
                    row['P bid'] = repr(quote.theo.b)
                else:
                    row['P bid'] = '-'
                if quote and quote.book[1]:
                    row['P ask.red'] = repr(quote.book[1])
                elif quote and quote.theo and quote.theo.a.a:
                    row['P ask'] = repr(quote.theo.a)
                else:
                    row['P ask'] = '-'
                row['P vol ask'] = f'{format_num(quote and quote.vols[1] * 100, 1)}%'
                result.append(row)
            result.append({})
        return result

    def count_open_orders(self):
        open_orders = 0
        for quote in self._quotes.values():
            open_orders += sum(1 for o in quote.book if o is not None)
        return open_orders

    async def try_read_ws_message(self, websocket):
        try:
            return json.loads(await asyncio.wait_for(websocket.recv(), timeout=1))
        except asyncio.TimeoutError:
            return None

    async def websocket_handler(self, websocket):
        await websocket.send(json.dumps({'min_delta': self.cfg.min_delta}))
        await websocket.send(json.dumps({'max_delta': self.cfg.max_delta}))
        try:
            while True:
                await websocket.send(json.dumps({'table_data': self.get_instrument_table_data()}))
                await websocket.send(json.dumps({'armed': self._armed}))
                await websocket.send(json.dumps({'open_orders': self.count_open_orders()}))
                if message := await self.try_read_ws_message(websocket):
                    match message.get('type'):
                        case 'armed_checkbox':
                            self._armed = message['status']
                            logging.info(f'set {self._armed}')
                        case 'min_delta':
                            self.cfg.min_delta = float(message['value'])
                        case 'max_delta':
                            self.cfg.max_delta = float(message['value'])
        except websockets.exceptions.ConnectionClosed:
            logging.info('Client connection closed')
            
    async def run_websocket_server(self):
        await websockets.serve(self.websocket_handler, "localhost", 8501)


def format_num(num, precision):
    if num is None:
        return '-'
    return f'{num:.{precision}f}'


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    run = True  # We set this to false when we want to stop
    cfg: Settings = default_settings()
    while run:
        iv_store = {}
        d = Deribit(iv_store, DERIBIT_URL, UNDERLYING)
        thalex = th.Thalex(network=NETWORK)
        q = Quoter(iv_store, thalex, cfg)
        app = web.Application()
        app.router.add_get('/', handle_http)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8500)
        await site.start()
        logging.info("http server started")
        tasks = [
            asyncio.create_task(d.task()),
            asyncio.create_task(q.read_task()),
            asyncio.create_task(q.send_task()),
            asyncio.create_task(q.update_task()),
            asyncio.create_task(q.run_websocket_server()),
        ]
        try:
            logging.info(f"STARTING on {NETWORK} {UNDERLYING=}")
            await asyncio.gather(*tasks)
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Lost connection ({e}). Reconnecting...")
            time.sleep(0.1)
        except asyncio.CancelledError:
            run = False
        except:
            logging.exception("There was an unexpected error:")
            time.sleep(0.5)
        await runner.cleanup()
        if thalex.connected():
            await thalex.cancel_session(id=CID_CANCEL_SESSION)
            while True:
                r = await thalex.receive()
                r = json.loads(r)
                if r.get("id", -1) == CID_CANCEL_SESSION:
                    logging.info(f"Cancelled session orders")
                    break
            await thalex.disconnect()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    asyncio.run(main())
