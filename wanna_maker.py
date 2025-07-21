import asyncio
import datetime
import json
import logging
import socket
import math
import time

import websockets

import keys
import thalex as th

from common import *
from deribit import Deribit

MAX_DTE = 7
MIN_DELTA = 0.2
MAX_DELTA = 0.8
LOTS = 0.1
BATCH = 100  # Mass Quote batch size
DISCO_SECS = 10  # cancel on disconnect seconds
MMP_SIZE = 3 * LOTS
W_ASK = 100  # $
W_BID = 100  # $
MIN_POS = -0.1
MAX_POS = 0.1

AMEND_THRESHOLD = 15  # $USD

DERIBIT_URL = "wss://www.deribit.com/ws/api/v2"
NETWORK = th.Network.DEV
UNDERLYING = "BTC"
PRODUCT = "OBTCUSD"
assert UNDERLYING in PRODUCT


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
        if (self.book[0] is None and self.theo.b is not None) or (self.book[1] is None and self.theo.a is not None):
            return True
        assert self.theo.b is not None
        assert self.theo.a is not None
        if not self.theo.b.p - AMEND_THRESHOLD < self.book[0].p < self.theo.b.p + AMEND_THRESHOLD:
            return True
        if not self.theo.a.p - AMEND_THRESHOLD < self.book[1].p < self.theo.a.p + AMEND_THRESHOLD:
            return True
        if self.book[1].a < self.theo.a.a:
            return True
        return False

    def update_theo(self, index: float, now: float, pp: float):
        logging.debug(f"{index=} {self.instrument.name} {self.delta=} {self.fwd=} {self.vols=}")
        if self.delta is None or self.fwd is None or not MIN_DELTA < abs(self.delta) < MAX_DELTA:
            self.theo.b = th.SideQuote(0, 0)
            self.theo.a = th.SideQuote(0, 0)
            return
        tte = (self.instrument.exp - now) / (3600 * 24 * 365.25)
        if self.instrument.type == InstrumentType.CALL:
            if pp < MAX_POS:
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, self.vols[0], tte)) - W_BID
                self.theo.b = th.SideQuote(p, LOTS if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, self.vols[1], tte)) + W_ASK
                self.theo.a = th.SideQuote(p, LOTS if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        elif self.instrument.type == InstrumentType.PUT:
            if pp < MAX_POS:
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, self.vols[0], tte)) - W_BID
                self.theo.b = th.SideQuote(p, LOTS if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, self.vols[1], tte)) + W_ASK
                self.theo.a = th.SideQuote(p, LOTS if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        logging.debug(f"{index=} {self.fwd=} {self.vols=} {self.instrument.name} theo: {self.theo}")


class Quoter:
    def __init__(self, iv_store: IvStore, thalex: th.Thalex):
        self.thalex = thalex
        self._iv_store: IvStore = iv_store
        self._instruments = {}
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
        await self.thalex.set_mm_protection(PRODUCT, MMP_SIZE, MMP_SIZE, id=CID_MMP)
        await self.thalex.instruments(CID_INSTRUMENTS)
        await self.thalex.public_subscribe([f"price_index.{UNDERLYING}USD"], CID_SUBSCRIBE)
        await self.thalex.private_subscribe(["session.orders", "account.portfolio"], id=CID_SUBSCRIBE)
        self._armed = True
        while True:
            msg = json.loads(await self.thalex.receive())
            cid = msg.get("id", CID_IGNORE)
            if cid == CID_INSTRUMENTS:
                subs = self.proc_instruments(msg["result"])
                await self.thalex.public_subscribe(subs, CID_SUBSCRIBE)
            elif cid == CID_SUBSCRIBE:
                logging.info(f"sub result: {msg}")
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
                    q.book[side] = th.SideQuote(0, 0)
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
        while True:
            await asyncio.sleep(0.25)
            queue = self._send_queue
            self._send_queue = []
            if not self._armed:
                continue
            logging.debug(f"sending {len(queue)} quotes")
            for i in range(0, len(queue), BATCH):
                batch = queue[i:i + BATCH]
                quotes = [el.theo for el in batch]
                for el in batch:
                    el.in_flight = True
                    el.queued = False
                await self.thalex.mass_quote(quotes, post_only=True, id=CID_QUOTE)

    def proc_instruments(self, instruments):
        now = datetime.datetime.now(datetime.UTC).timestamp()
        subs = []
        for i in instruments:
            expiry = i.get("expiration_timestamp", now + MAX_DTE + 50000)
            if i["product"] == PRODUCT and i["type"] == "option" and expiry < now + MAX_DTE * 24 * 3600:
                i = Instrument(
                    name=i["instrument_name"],
                    expiry=expiry,
                    itype=InstrumentType(i["option_type"]),
                    k=i["strike_price"]
                )
                if i.exp not in self._instruments:
                    self._instruments[i.exp] = {}
                if i.k not in self._instruments[i.exp]:
                    self._instruments[i.exp][i.k] = {}
                self._instruments[i.exp][i.k][i.type] = i.name
                self._quotes[i.name] = QuoteMeta(i)
                subs.append(f"ticker.{i.name}.500ms")
        return subs

    def update_vols(self):
        now = datetime.datetime.now(datetime.UTC).timestamp()
        nothave = []
        for exp, chain in self._instruments.items():
            iv_chain = self._iv_store.get(exp)
            if iv_chain is None:
                if exp not in nothave:
                    nothave.append(exp)
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
                        bid_iv_down = ndown[pc].bid_iv
                        bid_iv_up = nup[pc].bid_iv
                        ask_iv_down = ndown[pc].ask_iv
                        ask_iv_up = nup[pc].ask_iv
                        bid = (bid_iv_down + bid_iv_up) / 2
                        ask = (ask_iv_down + ask_iv_up) / 2
                        q = self._quotes[iname]
                        q.vols[0] = bid
                        q.vols[1] = ask
                        if self._index is None:
                            continue
                        pp = self.portfolio.get(q.instrument.name, 0)
                        q.update_theo(self._index, now, pp)
                        if q.should_send():
                            q.queued = True
                            self._send_queue.append(q)
        logging.debug(f"Deribit doesn't have expiries: {nothave}")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    run = True  # We set this to false when we want to stop
    while run:
        iv_store = {}
        d = Deribit(iv_store, DERIBIT_URL, UNDERLYING)
        thalex = th.Thalex(network=NETWORK)
        q = Quoter(iv_store, thalex)
        tasks = [
            asyncio.create_task(d.task()),
            asyncio.create_task(q.read_task()),
            asyncio.create_task(q.send_task()),
            asyncio.create_task(q.update_task()),
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

asyncio.run(main())
