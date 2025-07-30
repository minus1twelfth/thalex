import asyncio
import json
import logging

import thalex as th
from thalex import SideQuote

import keys
from blackscholes import call_discount, put_discount
from common import *
from settings import Settings

BATCH = 100  # Mass Quote batch size
DISCO_SECS = 10  # cancel on disconnect seconds
MIN_POS = -0.1
MAX_POS = 0.1

AMEND_THRESHOLD = 15  # $USD


def round_to_tick(value):
    tick = 5
    return tick * round(value / tick)


def width_discount(width_usd: float, delta: float):
    delta = abs(delta)
    if delta > 0.6:
        return width_usd
    elif delta < 0.1:
        return width_usd * 0.1
    else:
        # linear increase from delta 0.1 to 0.6
        return 0.1 * width_usd + (delta - 0.1) * width_usd * 0.9 / 0.5


def iv_offset_for_delta(vol_offsets: list[float], delta: float):
    assert delta > 0
    if delta <= 0.5:
        return vol_offsets[0] + (vol_offsets[1] - vol_offsets[0]) * 4 * max(delta - 0.25, 0)
    else:
        return vol_offsets[1] + (vol_offsets[2] - vol_offsets[1]) * 4 * min(delta - 0.5, 0.25)


def quote_needs_update(book: Optional[SideQuote], theo: SideQuote):
    assert theo is not None
    if book is None:
        return theo.a > 0  # insert
    if theo.a > 0:
        return not theo.p - AMEND_THRESHOLD < book.p < theo.p + AMEND_THRESHOLD  # amend
    else:
        return True  # delete


class QuoteMeta:
    def __init__(self, instrument: Instrument):
        self.theo = th.Quote(instrument.name, None, None)
        self.delta: Optional[float] = None
        self.fwd: Optional[float] = None
        self.book: list[Optional[th.SideQuote]] = [None, None]  # bid, ask
        self.vols: list[Optional[float]] = [None, None]  # bid ask
        self.iv_off: float = 0
        self.in_flight: bool = False
        self.queued: bool = False
        self.instrument: Instrument = instrument

    def should_send(self) -> bool:
        if self.in_flight or self.queued:
            return False
        return quote_needs_update(self.book[0], self.theo.b) or quote_needs_update(self.book[1], self.theo.a)

    def clear_theo(self):
        self.theo.b = th.SideQuote(0, 0)
        self.theo.a = th.SideQuote(0, 0)

    def update_theo(self, index: float, now: float, pp: float, cfg: Settings):
        logging.debug(f"{index=} {self.instrument.name} {self.delta=} {self.fwd=} {self.vols=}")
        if self.delta is None or self.fwd is None or not cfg.min_delta < abs(self.delta) < cfg.max_delta:
            self.theo.b = th.SideQuote(0, 0)
            self.theo.a = th.SideQuote(0, 0)
            return
        tte = (self.instrument.expiry.date.timestamp() - now) / (3600 * 24 * 365.25)
        vol_offsets = cfg.vol_offsets.get(str(self.instrument.expiry), [0, 0, 0])
        if self.instrument.type == InstrumentType.CALL:
            self.iv_off = iv_offset_for_delta(vol_offsets, self.delta)
            if pp < MAX_POS:
                iv = self.vols[0] + self.iv_off
                width = width_discount(cfg.width_bid_call, self.delta)
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, iv, tte) - width)
                self.theo.b = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                iv = self.vols[1] + self.iv_off
                width = width_discount(cfg.width_ask_call, self.delta)
                p = round_to_tick(call_discount(self.fwd, self.instrument.k, iv, tte) + width)
                self.theo.a = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        elif self.instrument.type == InstrumentType.PUT:
            self.iv_off = iv_offset_for_delta(vol_offsets, 1 - self.delta)
            if pp < MAX_POS:
                iv = self.vols[0] + self.iv_off
                width = width_discount(cfg.width_bid_put, self.delta)
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, iv, tte) - width)
                self.theo.b = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.b = th.SideQuote(0, 0)
            if pp > MIN_POS:
                iv = self.vols[1] + self.iv_off
                width = width_discount(cfg.width_ask_put, self.delta)
                p = round_to_tick(put_discount(self.fwd, self.instrument.k, iv, tte) + width)
                self.theo.a = th.SideQuote(p, cfg.quote_size if p > 10 else 0)
            else:
                self.theo.a = th.SideQuote(0, 0)
        logging.debug(f"{index=} {self.fwd=} {self.vols=} {self.instrument.name} theo: {self.theo}")


class Quoter:
    def __init__(
            self,
            iv_store: IvStore,
            thalex: th.Thalex,
            cfg: Settings,
            network: th.Network,
            underlying: str,
            product: str
    ):
        self.thalex = thalex
        self._network = network
        self._underlying = underlying
        self._product = product
        self._cfg = cfg
        self._iv_store: IvStore = iv_store
        self._instruments: dict[Expiry, dict[float, dict[InstrumentType, str]]] = {}
        self._quotes: dict[str, QuoteMeta] = {}
        self._send_queue: list[QuoteMeta] = []
        self._index: Optional[float] = None
        self._armed: bool = False
        self._portfolio: dict[str, float] = {}

    @property
    def cfg(self):
        return self._cfg

    @property
    def instruments(self):
        return self._instruments

    @property
    def armed(self):
        return self._armed

    @armed.setter
    def armed(self, armed):
        self._armed = armed
        logging.info(f'{"" if self._armed else "dis"}arming quoter')

    @property
    def index(self):
        return self._index

    @property
    def quotes(self):
        return self._quotes

    @property
    def portfolio(self):
        return self._portfolio

    async def read_task(self):
        await self.thalex.connect()
        await self.thalex.login(keys.key_ids[self._network], keys.private_keys[self._network], id=CID_LOGIN, )
        await self.thalex.set_cancel_on_disconnect(DISCO_SECS, CID_CANCEL_DISCO)
        await self.thalex.instruments(CID_INSTRUMENTS)
        await self.thalex.public_subscribe([f"price_index.{self._underlying}USD"], CID_SUBSCRIBE)
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
            elif cid in [CID_MMP, CID_LOGIN, CID_CANCEL_DISCO]:
                logging.debug(msg)
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
            for o in n:
                if o.get("delete_reason", "") == "mm_protection" and self._armed:
                    logging.warning("MMP is hit. Disarming.")
                    self._armed = False
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
                self._portfolio[pp["instrument_name"]] = pp["position"]

    async def send_task(self):
        is_quoting = False
        while True:
            await asyncio.sleep(0.25)
            if not self._armed:
                if is_quoting:
                    await self.thalex.cancel_mass_quote(id=CID_MMP)
                    is_quoting = False
                continue
            elif not is_quoting:
                await self.thalex.set_mm_protection(self._product, self._cfg.mmp_size, self._cfg.mmp_size, id=CID_MMP)
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
                await self.thalex.mass_quote(quotes, post_only=True, id=CID_QUOTE, label=self._cfg.label)

    def proc_instruments(self, instruments):
        subs = []
        for i in instruments:
            if i["product"] == self._product and i["type"] == "option":
                i = Instrument(name=i["instrument_name"], expiry=Expiry(i["expiration_timestamp"]),
                               itype=InstrumentType(i["option_type"]), k=i["strike_price"])
                exp_instruments = self._instruments.setdefault(i.expiry, {})
                strike_instruments = exp_instruments.setdefault(i.k, {})
                strike_instruments[i.type] = i.name
                self._quotes[i.name] = QuoteMeta(i)
                subs.append(f"ticker.{i.name}.500ms")
        return subs

    def update_vols(self):
        now = datetime.datetime.now(datetime.UTC).timestamp()
        nothave = set()
        for expiry, chain in self._instruments.items():
            iv_chain = self._iv_store.get(expiry)
            if iv_chain is None:
                nothave.add(expiry)
                continue
            expiry_is_enabled = self._cfg.expiry_is_enabled(str(expiry))

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
                        bid = (bid_iv_down * (k_up - k) + bid_iv_up * (k - k_down)) / k_diff
                        ask = (ask_iv_down * (k_up - k) + ask_iv_up * (k - k_down)) / k_diff
                    q.vols[0] = bid
                    q.vols[1] = ask
                    if self._index is None:
                        continue
                    pp = self._portfolio.get(q.instrument.name, 0)
                    if expiry_is_enabled:
                        q.update_theo(self._index, now, pp, self._cfg)
                    else:
                        q.clear_theo()
                    if q.should_send() and self._armed:
                        q.queued = True
                        self._send_queue.append(q)

        logging.debug(f"Deribit doesn't have expiries: {nothave}")

    def count_open_orders(self):
        open_orders = 0
        for quote in self._quotes.values():
            open_orders += sum(1 for o in quote.book if o is not None)
        return open_orders
