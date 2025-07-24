import json
from datetime import datetime

import websockets
import logging

from common import *


class Deribit:
    def __init__(self, iv_store: IvStore, url: str, underlying: str):
        self._underlying = underlying
        self._url = url
        self._ws = None
        self._instruments: dict[str, Instrument] = {}
        self._iv_store: IvStore = iv_store

    async def connect(self):
        self._ws = await websockets.connect(self._url, ping_interval=5)

    async def subscribe(self, channels):
        await self._send("public/subscribe", {"channels": channels}, CID_SUBSCRIBE)

    # currency: BTC; ETH; USDC; USDT; EURR; any
    # kind: future; option; spot; future_combo; option_combo; None=all
    async def instruments(self, currency, kind=None):
        await self._send("/public/get_instruments", {"currency": currency, "kind": kind}, CID_INSTRUMENTS)

    async def disconnect(self):
        await self._ws.close()

    async def _send(self, method, params, cid):
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": cid
        })
        await self._ws.send(msg)

    async def recv(self):
        return json.loads(await self._ws.recv())

    def proc_instruments(self, instruments):
        subs = []
        for i in instruments:
            assert i["kind"] == "option"  # only options are supported for now
            expiry = i["expiration_timestamp"] / 1000
            i = Instrument(
                name=i["instrument_name"],
                expiry=Expiry(expiry),
                itype=InstrumentType(i["option_type"]),
                k=i["strike"]
            )
            self._instruments[i.name] = i
            subs.append(f"ticker.{i.name}.100ms")
        return subs

    def proc_notification(self, n):
        ch = n["channel"]
        d = n["data"]
        if ch.startswith("ticker"):
            i = self._instruments.get(d["instrument_name"])
            t = Ticker(bid_iv=d["bid_iv"]/100, ask_iv=d["ask_iv"]/100, mark_iv=d["mark_iv"]/100, delta=d["greeks"]["delta"])
            if i is not None:
                iv_exp = self._iv_store.setdefault(i.expiry, {})
                iv_strike = iv_exp.setdefault(i.k, {})
                iv_strike[i.type] = t

    async def task(self):
        await self.connect()
        await self.instruments(self._underlying, "option")
        while True:
            msg = await self.recv()
            cid = msg.get("id", CID_IGNORE)
            method = msg.get("method", "")
            if cid == CID_INSTRUMENTS:
                subs = self.proc_instruments(msg["result"])
                await self.subscribe(subs)
            elif cid == CID_SUBSCRIBE:
                logging.debug(f"subscription result: {msg}")
                pass
            elif method == "subscription":
                self.proc_notification(msg["params"])
            else:
                logging.warning(f"Unhandled msg: {msg}")
