import asyncio
import html
import datetime
import json
import logging
import time
import traceback

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, filters, ContextTypes, CallbackQueryHandler

import thalex as th
import keys
from common import Expiry, tlx_instrument, Instrument, InstrumentType
import blackscholes as bs

CID_PORTFOLIO = 1
CID_ACC_SUM = 2
CID_INSTRUMENTS = 3
CID_TRADE_HISTORY = 4
CID_TICK_START = 100
CID_TICK_END = 500

NETWORK = th.Network.PROD
COOLDOWN = 3600 * 4

SECS_IN_YEAR = 3600.0 * 24.0 * 365.25

SCENARIOS = [(0, 0), (0.02, 0.05), (0.02, -0.05), (-0.02, 0.05), (-0.02, -0.05)]  # (+index, +iv)


def load_persisted_data() -> dict:
    try:
        with open('tg_bot.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def persist_data(data: dict):
    with open('tg_bot.json', 'w') as file:
        json.dump(data, file, indent=2)


class AccountSummary:
    def __init__(self, cash, balance, req, upnl, rpnl, im, mm):
        self.cash = cash
        self.balance = balance
        self.req = req
        self.upnl = upnl
        self.rpnl = rpnl
        self.im = im
        self.mm = mm


class UnderlyingGreeks:
    def __init__(self):
        self.delta = 0
        self.delta_cash = 0
        self.gamma = 0
        self.vega = 0
        self.theta = 0

    def __repr__(self):
        return (f"{'Δ':<5} |   {self.delta:.2f}\n"
                f"{'$Δ':<5} | $ {self.delta_cash:.0f}\n"
                f"{'Γ':<5} |   {self.gamma:.4f}\n"
                f"{'Θ':<5} | $ {self.theta:.0f}\n"
                f"{'ν':<5} | $ {self.vega:.0f}\n")

    def take(self, pp, tick, i: Instrument, now: datetime.datetime, index, fwd_off=0.0, iv_off=0.0):
        if i.type in [InstrumentType.CALL, InstrumentType.PUT]:
            tte = (i.expiry.date - now).total_seconds() / SECS_IN_YEAR
            fwd = tick["forward"] * (1+fwd_off)
            sigma = tick["iv"] + iv_off
            idelta = bs.call_delta(fwd, i.k, sigma, tte) if i.type == InstrumentType.CALL else bs.put_delta(fwd, i.k, sigma, tte)
            d = pp * idelta
            self.delta += d
            self.delta_cash += d * index
            self.gamma += pp * bs.gamma(fwd, i.k, sigma, tte)
            self.theta += pp * bs.theta(fwd, i.k, sigma, tte)
            self.vega += pp * bs.vega(fwd, i.k, sigma, tte)
        else:
            self.delta += pp
            self.delta_cash += pp * index


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
            f"{'BTC':<5} | $ {self.i_btc:.0f}\n"
            f"{'-'*25}\n"
            f"{self.btc}"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'ETH':<5} | $ {self.i_eth:.0f}\n"
            f"{'-'*25}\n"
            f"{self.eth}"
            f"{' '*25}\n"
            f"{' '*25}\n"
            f"{'Σ$Δ':<5} | $ {self.eth.delta_cash + self.btc.delta_cash:.0f}\n"
            "</pre>"
        )

async def connect_to_exchange() -> th.Thalex:
    thalex = th.Thalex(network=NETWORK)
    await thalex.connect()
    await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
    return thalex

async def get_portfolio(thalex: th.Thalex):
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
                return positions, tickers, instruments
        elif cid == CID_INSTRUMENTS:
            instruments = [tlx_instrument(i) for i in msg["result"]]
            instruments = {i.name: i for i in instruments}


async def get_account_summary(thalex: th.Thalex):
    await thalex.account_summary(id=CID_ACC_SUM)
    while True:
        msg = json.loads(await thalex.receive())
        if msg["id"] == CID_ACC_SUM:
            msg = msg["result"]
            bal = msg["margin"]
            req = msg["required_margin"]
            im = 100 * req/bal
            mm = im * 0.7
            return AccountSummary(
                cash=int(msg['cash_collateral']),
                balance=int(bal),
                req=int(req),
                upnl=int(msg['unrealised_pnl']),
                rpnl=int(msg['session_realised_pnl']),
                im=im,
                mm=mm
            )


async def get_recent_trades(thalex: th.Thalex):
    pd = load_persisted_data()
    last_trade_ts = pd.get("last_trade_ts", 0)
    await thalex.trade_history(time_low=last_trade_ts, id=CID_TRADE_HISTORY)
    while True:
        msg = json.loads(await thalex.receive())
        if msg["id"] == CID_TRADE_HISTORY:
            msg = msg["result"]
            trades = []
            for t in msg["trades"]:
                if t["trade_type"] == "expiration":
                    continue
                if not t["time"] > last_trade_ts:
                    continue
                trades.append(t)
            if trades:
                pd["last_trade_ts"] = max(t["time"] for t in trades)
                persist_data(pd)
            return trades



async def margin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thalex = await connect_to_exchange()
    s = await get_account_summary(thalex)
    msg = (
        "<pre>"
        f"{'Cash':<10} | $ {s.cash}\n"
        f"{'Balance':<10} | $ {s.balance}\n"
        f"{'Req':<10} | $ {s.req}\n"
        f"{'UPnL':<10} | $ {s.upnl}\n"
        f"{'RPnL':<10} | $ {s.rpnl}\n"
        f"{'IM':<10} | % {s.im:.0f}\n"
        f"{'MM':<10} | % {s.mm:.0f}\n"
        "</pre>"
    )
    await context.bot.send_message(keys.CHAT_ID, msg, parse_mode='HTML')
    await thalex.disconnect()


async def greeks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thalex = await connect_to_exchange()
    portfolio, tickers, instruments = await get_portfolio(thalex)
    g = Greeks(portfolio, tickers, instruments)
    await context.bot.send_message(keys.CHAT_ID, f"{g}", parse_mode='HTML')
    await thalex.disconnect()


async def scenarios(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("BTC", callback_data="BTCUSD"),
            InlineKeyboardButton("ETH", callback_data="ETHUSD"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(keys.CHAT_ID, "Choose an underlying", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data not in ["BTCUSD", "ETHUSD"]:
        await query.edit_message_text(text=f"I don't know that underlying")
        return
    thalex = await connect_to_exchange()
    portfolio, tickers, instruments = await get_portfolio(thalex)
    outcomes = {}
    for s in SCENARIOS:
        s_greeks = UnderlyingGreeks()
        now = datetime.datetime.now(tz=datetime.timezone.utc)

        for iname, pp in portfolio.items():
            tick = tickers[iname]
            i = instruments[iname]
            index = tick["index"] * (1+s[0])
            if instruments[iname].underlying == f"{query.data}":
                s_greeks.take(pp, tick, i, now, index, fwd_off=s[0], iv_off=s[1])
        outcomes[s] = s_greeks

    msg = ""
    for s in SCENARIOS:
        msg += f"<b>{query.data}: {'+' if s[0] > 0 else ''}{s[0] * 100:.0f}%, iv: {s[1] * 100:.0f}%</b>"
        msg += "<pre>"
        msg += f"{outcomes[s]}"
        msg += "</pre>\n\n"
    await query.edit_message_text(text=msg, parse_mode='HTML')
    await thalex.disconnect()


async def h_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thalex = await connect_to_exchange()
    portfolio, tickers, instruments = await get_portfolio(thalex)
    next_exp = min([instruments[i] for i in portfolio.keys()], key=lambda i: i.expiry).expiry
    remaining = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry != next_exp}
    expiring = {iname: pp for iname, pp in portfolio.items() if instruments[iname].expiry == next_exp}
    e_greek = Greeks(expiring, tickers, instruments)
    r_greek = Greeks(remaining, tickers, instruments)
    exp_pos_str = "<pre>"
    for iname, pp in expiring.items():
        if "BTC" in iname:
            exp_pos_str += f"{iname:<20} | {pp}\n"
    exp_pos_str += f"{' '*30}\n"
    for iname, pp in expiring.items():
        if "ETH" in iname:
            exp_pos_str += f"{iname:<20} | {pp}\n"
    exp_pos_str += "</pre>"
    await context.bot.send_message(keys.CHAT_ID, f"The next expiry on <b>{next_exp}</b> "
                                    f"will see the following positions and greeks expiring:\n"
                                    f"{exp_pos_str}\n"
                                    f"{e_greek}\n\n\n"
                                    f"After that, greeks will look like:\n"
                                    f"{r_greek}", parse_mode='HTML')
    await thalex.disconnect()

def format_error_message(error):
    error_lines = traceback.format_exception(error)
    return f'Unexpected error <pre>{html.escape("".join(error_lines))}</pre>'

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)
    await context.bot.send_message(chat_id=keys.ADMIN_ID, text=format_error_message(context.error), parse_mode='HTML')
    await context.bot.send_message(chat_id=keys.CHAT_ID, text='An unexpected error has occurred')


class Alert:
    def __init__(self, name, base_lvl, inc, eval):
        self.name = name
        self.base_lvl = base_lvl
        self.cur_lvl = base_lvl
        self.inc = inc
        self.last_report = 0
        self.eval = eval

    async def check_notify(self, app, now, greeks, acc_sum):
        sig = self.eval(greeks, acc_sum)
        notify = False
        while sig > self.cur_lvl + self.inc:
            self.cur_lvl += self.inc
            notify = True
        notify |= sig > self.cur_lvl and now > self.last_report + COOLDOWN
        if sig < self.base_lvl * 0.5:
            self.cur_lvl = self.base_lvl
            self.last_report = 0
        if notify:
            text = f'⚠️<b>ALERT: {self.name}</b> value <b>{sig:.0f}</b>'
            await app.bot.send_message(chat_id=keys.CHAT_ID, text=text, parse_mode='HTML')
            self.last_report = now


async def check_greeks_forever(app):
    last_error_time = 0
    alerts = [
        Alert("BTC $Δ", 3000, 500, lambda g, s: abs(g.btc.delta_cash)),
        Alert("ETH $Δ", 3000, 500, lambda g, s: abs(g.eth.delta_cash)),
        Alert("Σ$Δ", 2000, 500, lambda g, s: abs(g.eth.delta_cash + g.btc.delta_cash)),
        Alert("Margin", 75, 5, lambda g, s: s.im),
        Alert("Session Loss", 1000, 500, lambda g, s: -s.upnl - s.rpnl),
    ]
    while True:
        now = time.time()
        try:
            thalex = await connect_to_exchange()
            portfolio, tickers, instruments = await get_portfolio(thalex)
            g = Greeks(portfolio, tickers, instruments)
            s = await get_account_summary(thalex)
            for a in alerts:
                await a.check_notify(app, now, g, s)
            trades = await get_recent_trades(thalex)
            for t in trades:
                direction = "Bought" if t["direction"] == "buy" else "Sold"
                msg = (f"{direction} {t['amount']}@{int(t['price'])} in {t['instrument_name']}"
                       f", position is {t['position_after']}")
                await app.bot.send_message(chat_id=keys.CHAT_ID, text=msg)
            await thalex.disconnect()
        except Exception as e:
            if now > last_error_time + 30*60:
                await app.bot.send_message(chat_id=keys.ADMIN_ID, text=format_error_message(e), parse_mode='HTML')
                last_error_time = now
        await asyncio.sleep(60)



async def post_init(app):
    commands = [
        BotCommand("greeks", "Report greeks per underlying"),
        BotCommand("scenarios", "Report greeks in a set of scenarios"),
        BotCommand("next", "Report positions that expire next, and what the greeks will be after"),
        BotCommand("margin", "Report margin use and PNL"),
    ]
    await app.bot.set_my_commands(commands)
    app.create_task(check_greeks_forever(app))

def main():
    app = ApplicationBuilder().token(keys.TG_TOKEN).post_init(post_init).build()
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
        "scenarios",
        scenarios,
        filters=filters.Chat(chat_id=keys.CHAT_ID)
    ))
    app.add_handler(CommandHandler(
        "next",
        h_next,
        filters=filters.Chat(chat_id=keys.CHAT_ID)
    ))
    app.add_handler(CallbackQueryHandler(button))
    app.add_error_handler(error_handler)
    app.run_polling()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    main()
