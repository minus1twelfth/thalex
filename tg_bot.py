import json
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

import thalex as th
import keys

CID_PORTFOLIO = 1
CID_ACC_SUM = 2
CID_TICK_START = 100
CID_TICK_END = 500


class Greeks:
    def __init__(self):
        self.d_btc = 0
        self.d_eth = 0
        self.d_cash = 0
        self.i_btc = -1
        self.i_eth = -1

    def __repr__(self):
        return (f"Cash Delta: ${self.d_cash:.0f}\nBTC:\n"
                f"\tindex: ${self.i_btc:.0f}\n"
                f"\tdelta: {self.d_btc:.2f}\n"
                f"ETH:\n"
                f"\tindex: ${self.i_eth:.0f}\n"
                f"\tdelta: {self.d_eth:.2f}")


async def get_greeks() -> Greeks:
    network = th.Network.PROD
    thalex = th.Thalex(network=network)
    await thalex.connect()
    await thalex.login(keys.key_ids[network], keys.private_keys[network])
    await thalex.portfolio(id=CID_PORTFOLIO)
    tickers = {}
    ticker_calls = {}
    positions = {}
    g = Greeks()
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
                for iname, pp in positions.items():
                    tick = tickers[iname]
                    d = tick["delta"]
                    index = tick["index"]
                    if "BTC" in iname:
                        g.d_btc += pp * d
                        g.d_cash += pp * d * index
                        g.i_btc = index
                    elif "ETH" in iname:
                        g.d_eth += pp * d
                        g.d_cash += pp * d * index
                        g.i_eth = index
                await thalex.disconnect()
                return g


async def get_margin():
    network = th.Network.PROD
    thalex = th.Thalex(network=network)
    await thalex.connect()
    await thalex.login(keys.key_ids[network], keys.private_keys[network])
    await thalex.account_summary(id=CID_ACC_SUM)
    while True:
        msg = json.loads(await thalex.receive())
        if msg["id"] == CID_ACC_SUM:
            msg = msg["result"]
            bal = msg["margin"]
            req = msg["required_margin"]
            im = req/bal
            mm = im * 0.7
            await thalex.disconnect()
            return (f"cash: ${int(msg["cash_collateral"])},\n"
                    f"balance: ${int(bal)},\n"
                    f"requirement: ${int(req)},\n"
                    f"upnl: ${int(msg["unrealised_pnl"])},\n"
                    f"rpnl: ${int(msg["session_realised_pnl"])}\n"
                    f"im: {im:.2f}   mm: {mm:.2f}")


async def margin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = await get_margin()
    await update.message.reply_text(m)


async def greeks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    g = await get_greeks()
    await update.message.reply_text(f"{g}")


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
    app.add_error_handler(error_handler)

    app.run_polling()


if __name__ == "__main__":
    main()