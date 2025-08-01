import asyncio
import datetime
import json
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, filters, ContextTypes

import thalex as th
import keys
from common import Expiry, tlx_instrument, Instrument, InstrumentType
import blackscholes as bs

CID_PORTFOLIO = 1
CID_ACC_SUM = 2
CID_INSTRUMENTS = 3
CID_TICK_START = 100
CID_TICK_END = 500

NETWORK = th.Network.PROD

SECS_IN_YEAR = 3600.0 * 24.0 * 365.25


class UnderlyingGreeks:
    def __init__(self):
        self.delta = 0
        self.delta_cash = 0
        self.gamma = 0
        self.vega = 0
        self.theta = 0

    def __repr__(self):
        return (f"{'Δ':<8} |   {self.delta:.2f}\n"
                f"{'Δ':<8} | $ {self.delta_cash:.0f}\n"
                f"{'gamma':<8} |   {self.gamma:.4f}\n"
                f"{'theta':<8} | $ {self.theta:.0f}\n"
                f"{'vega':<8} | $ {self.vega:.0f}\n")

    def take(self, pp, tick, i: Instrument, now: datetime.datetime, index):
        d = tick["delta"]
        self.delta += pp * d
        self.delta_cash += pp * d * index
        if i.type in [InstrumentType.CALL, InstrumentType.PUT]:
            tte = (i.expiry.date - now).total_seconds() / SECS_IN_YEAR
            fwd = tick["forward"]
            sigma = tick["iv"]
            self.gamma += pp * bs.gamma(fwd, i.k, sigma, tte)
            self.theta += pp * bs.theta(fwd, i.k, sigma, tte)
            self.vega += pp * bs.vega(fwd, i.k, sigma, tte)
            logging.info(f"{i.name} {fwd} {i.k} {sigma} {tte}  t: {pp * bs.theta(fwd, i.k, sigma, tte)}")


class Greeks:
    def __init__(self, portfolio, tickers, instruments):
        self.btc = UnderlyingGreeks()
        self.eth = UnderlyingGreeks()
        self.i_btc = -1
        self.i_eth = -1
        now = datetime.datetime.now(tz=datetime.timezone.utc)

        for iname, pp in portfolio.items():
            tick = tickers[iname]
            i = instruments[iname]
            index = tick["index"]
            if "BTC" in iname:
                self.btc.take(pp, tick, i, now, index)
                self.i_btc = index
            elif "ETH" in iname:
                self.eth.take(pp, tick, i, now, index)
                self.i_eth = index

    def __repr__(self):
        return (
            "<pre>"
            f"{'BTC':<8} | $ {self.i_btc:.0f}\n"
            f"{'-'*25}\n"
            f"{self.btc}"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'ETH':<8} | $ {self.i_eth:.0f}\n"
            f"{'-'*25}\n"
            f"{self.eth}"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'Tot $Δ':<8} | $ {self.eth.delta_cash + self.btc.delta_cash:.0f}\n"
            "</pre>"
        )


async def get_portfolio():
    thalex = th.Thalex(network=NETWORK)
    await thalex.connect()
    await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
    await thalex.portfolio(id=CID_PORTFOLIO)
    await thalex.instruments(id=CID_INSTRUMENTS)
    tickers = {}
    ticker_calls = {}
    positions = {}
    instruments = {}
    while True:
        msg = json.loads(await thalex.receive())
        cid = msg.get("id") or -1
        if cid == CID_PORTFOLIO:
            portfolio = msg["result"]
            assert len(portfolio) < CID_TICK_END - CID_TICK_START
            i = CID_TICK_START
            for pp in portfolio:
                iname = pp["instrument_name"]
                positions[iname] = pp["position"]
                ticker_calls[i] = iname
                await thalex.ticker(iname, id=i)
                i += 1
        elif CID_TICK_START <= cid < CID_TICK_END:
            iname = ticker_calls.pop(cid)
            tickers[iname] = msg["result"]
            if len(ticker_calls) == 0:
                await thalex.disconnect()
                return positions, tickers, instruments
        elif cid == CID_INSTRUMENTS:
            instruments = [tlx_instrument(i) for i in msg["result"]]
            instruments = {i.name: i for i in instruments}


async def get_greeks() -> Greeks:
    portfolio, tickers, instruments = await get_portfolio()
    return Greeks(portfolio, tickers, instruments)


async def get_next() -> (Expiry, dict, Greeks, Greeks):
    portfolio, tickers, instruments = await get_portfolio()
    next_exp = min([instruments[i] for i in portfolio.keys()], key=lambda i: i.expiry).expiry
    remaining = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry != next_exp}
    expiring = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry == next_exp}
    return next_exp, expiring, Greeks(expiring, tickers, instruments), Greeks(remaining, tickers, instruments)


async def get_margin():
    thalex = th.Thalex(network=NETWORK)
    await thalex.connect()
    await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
    await thalex.account_summary(id=CID_ACC_SUM)
    while True:
        msg = json.loads(await thalex.receive())
        if msg["id"] == CID_ACC_SUM:
            msg = msg["result"]
            bal = msg["margin"]
            req = msg["required_margin"]
            im = 100 * req/bal
            mm = im * 0.7
            await thalex.disconnect()
            return (
                "<pre>"
                f"{'Cash':<10} | $ {int(msg['cash_collateral'])}\n"
                f"{'Balance':<10} | $ {int(bal)}\n"
                f"{'Req':<10} | $ {int(req)}\n"
                f"{'UPnL':<10} | $ {int(msg['unrealised_pnl'])}\n"
                f"{'RPnL':<10} | $ {int(msg['session_realised_pnl'])}\n"
                f"{'IM':<10} | % {im:.0f}\n"
                f"{'MM':<10} | % {mm:.0f}\n"
                "</pre>"
            )


async def margin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = await get_margin()
    await update.message.reply_text(m, parse_mode='HTML')


async def greeks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    g = await get_greeks()
    await update.message.reply_text(f"{g}", parse_mode='HTML')


async def h_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    next_exp, exp_pp, e_greek, r_greek = await get_next()
    exp_pos_str = "<pre>"
    for iname, pp in exp_pp.items():
        if "BTC" in iname:
            exp_pos_str += f"{iname:<20} | {pp}\n"
    exp_pos_str += f"{' '*30}\n"
    for iname, pp in exp_pp.items():
        if "ETH" in iname:
            exp_pos_str += f"{iname:<20} | {pp}\n"
    exp_pos_str += "</pre>"
    await update.message.reply_text(f"The next expiry on <b>{next_exp}</b> "
                                    f"will see the following positions and greeks expiring:\n"
                                    f"{exp_pos_str}\n"
                                    f"{e_greek}\n\n\n"
                                    f"After that, greeks will look like:\n"
                                    f"{r_greek}", parse_mode='HTML')


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)
    if update and isinstance(update, Update):
        await update.message.reply_text("There was an oupsie processing this one. :(")

async def check_greeks_forever():
    while True:
        await asyncio.sleep(3)
        logging.info('checking greeks')

async def run_app():
    app = ApplicationBuilder().token(keys.TG_TOKEN).build()
    app.add_handler(CommandHandler(
        "margin",
        margin,
        filters=filters.Chat(chat_id=keys.CHAT_ID)
    ))
    app.add_handler(CommandHandler(
        "greeks",
        greeks,
        filters=filters.Chat(chat_id=keys.CHAT_ID)
    ))
    app.add_handler(CommandHandler(
        "next",
        h_next,
        filters=filters.Chat(chat_id=keys.CHAT_ID)
    ))
    app.add_error_handler(error_handler)
    forever = asyncio.Future()
    async with app:
        await app.start()
        await app.updater.start_polling()
        try:
            await forever
        except asyncio.CancelledError:
            pass
        await app.stop()


async def main():
    try:
        await asyncio.gather(run_app(), check_greeks_forever())
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    asyncio.run(main())