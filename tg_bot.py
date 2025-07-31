import json
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, filters, ContextTypes

import thalex as th
import keys
from common import Expiry, tlx_instrument

CID_PORTFOLIO = 1
CID_ACC_SUM = 2
CID_INSTRUMENTS = 3
CID_TICK_START = 100
CID_TICK_END = 500

NETWORK = th.Network.PROD


class Greeks:
    def __init__(self, portfolio, tickers):
        self.d_btc = 0
        self.d_btc_cash = 0
        self.d_eth = 0
        self.d_eth_cash = 0
        self.i_btc = -1
        self.i_eth = -1

        for iname, pp in portfolio.items():
            tick = tickers[iname]
            d = tick["delta"]
            index = tick["index"]
            if "BTC" in iname:
                self.d_btc += pp * d
                self.d_btc_cash += pp * d * index
                self.i_btc = index
            elif "ETH" in iname:
                self.d_eth += pp * d
                self.d_eth_cash += pp * d * index
                self.i_eth = index

    def __repr__(self):
        return (
            "<pre>"
            f"{'BTC':<8} | $ {self.i_btc:.0f}\n"
            f"{'-'*25}\n"
            f"{'Δ':<8} |   {self.d_btc:.2f}\n"
            f"{'Δ':<8} | $ {self.d_btc_cash:.0f}\n"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'ETH':<8} | $ {self.i_eth:.0f}\n"
            f"{'-'*25}\n"
            f"{'Δ':<8} |   {self.d_eth:.2f}\n"
            f"{'Δ':<8} | $ {self.d_eth_cash:.0f}\n"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'Tot $Δ':<8} | $ {self.d_eth_cash + self.d_btc_cash:.0f}\n"
            "</pre>"
        )


async def get_portfolio(thalex: th.Thalex):
    await thalex.portfolio(id=CID_PORTFOLIO)
    tickers = {}
    ticker_calls = {}
    positions = {}
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
                return positions, tickers


async def get_greeks() -> Greeks:
    thalex = th.Thalex(network=NETWORK)
    await thalex.connect()
    await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
    portfolio, tickers = await get_portfolio(thalex)
    await thalex.disconnect()
    return Greeks(portfolio, tickers)


async def get_next() -> (Expiry, dict, Greeks, Greeks):
    thalex = th.Thalex(network=NETWORK)
    await thalex.connect()
    await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
    portfolio, tickers = await get_portfolio(thalex)
    await thalex.instruments(id=CID_INSTRUMENTS)
    while True:
        msg = json.loads(await thalex.receive())
        cid = msg.get("id") or -1
        if cid == CID_INSTRUMENTS:
            instruments = [tlx_instrument(i) for i in msg["result"]]
            next_exp = min(instruments, key=lambda i: i.expiry).expiry
            instruments = {i.name: i for i in instruments}
            remaining = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry != next_exp}
            expiring = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry == next_exp}
            await thalex.disconnect()
            return next_exp, expiring, Greeks(expiring, tickers), Greeks(remaining, tickers)


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
    await update.message.reply_text(f"The next expiry is <b>{next_exp}</b>.\n"
                                    f"It'll see the following positions and greeks expiring:\n"
                                    f"{exp_pos_str}\n"
                                    f"{e_greek}\n\n\n"
                                    f"After that, greeks will look like:\n"
                                    f"{r_greek}", parse_mode='HTML')


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)
    if update and isinstance(update, Update):
        await update.message.reply_text("There was an oupsie processing this one. :(")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
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

    app.run_polling()


if __name__ == "__main__":
    main()