import asyncio
import json
import logging
import socket
import sys
import time
from typing import Optional
import websockets

import thalex
from thalex.thalex import Direction
import keys  # Rename _keys.py to keys.py and add your keys. There are instructions how to create keys in that file.


def fr_to_annualized(fr):
    return fr * 365.25 * 3


def annualized_to_fr(ann):
    return ann / (365.25 * 3)


def fr_to_price(fr, index):
    if fr > 0:
        return (1.00025 + fr) * index
    if fr == 0:
        return index
    return (0.99975 + fr) * index


# static cfg
ORDER_LABEL = "rebalancer"
INSTRUMENT = "BTC-PERPETUAL"
INDEX = "BTCUSD"
PRICE_TICK = 1
SIZE_TICK = 0.001
QUOTE_ID = {Direction.BUY: 1001, Direction.SELL: 1002}
EXIT_QUOTE_ID = 1003

# Runtime cfg defaults
MAX_POSITION = 0.04
MIN_POSITION = -0.04
TGT_POSITION = 0
SIZE = 0.005  # Number of contracts to quote
AMEND_THRESHOLD = 5  # USD
THEO_FR = annualized_to_fr(10 / 100)
ASK_OFFSET = 30  # $
BID_OFFSET = 30  # $
EXIT_OFFSET = 0  # $


class Config:
    def __init__(self):
        self.min_pos = MIN_POSITION
        self.max_pos = MAX_POSITION
        self.tgt_pos = TGT_POSITION
        self.size = SIZE
        self.amend_threshold = AMEND_THRESHOLD
        self.theo_fr = THEO_FR
        self.ask_offset = ASK_OFFSET
        self.bid_offset = BID_OFFSET
        self.exit_offset = EXIT_OFFSET

    def to_dict(self):
        return {
            "min_pos": self.min_pos,
            "max_pos": self.max_pos,
            "tgt_pos": self.tgt_pos,
            "size": self.size,
            "amend_threshold": self.amend_threshold,
            "theo_fr": fr_to_annualized(self.theo_fr) * 100,
            "bid_offset": self.bid_offset,
            "ask_offset": self.ask_offset,
            "x_offset": self.exit_offset,
        }

    def set(self, key, val):
        if not isinstance(key, str):
            return "invalid key"
        if not isinstance(val, float):
            return "invalid value"
        if key == "min_pos":
            if -0.5 <= val <= 0.5:
                self.min_pos = val
            else:
                return "min_pos must be between -0.5 and 0.5"
        elif key == "max_pos":
            if -0.5 <= val <= 0.5:
                self.max_pos = val
            else:
                return "max_pos must be between -0.5 and 0.5"
        elif key == "tgt_pos":
            if max(-0.5, self.min_pos) <= val <= min(0.5, self.max_pos):
                self.tgt_pos = val
            else:
                return "tgt_pos must be between max(-0.5, min_pos) and min(0.5, self.max_pos)"
        elif key == "size":
            if 0.001 <= val <= 0.05:
                self.size = val
            else:
                return "size must be between 0.001 and 0.05"
        elif key == "amend_threshold":
            if 3 <= val <= 10:
                self.amend_threshold = val
            else:
                return "amend_threshold must be between 3 and 10"
        elif key == "theo_fr":
            val = annualized_to_fr(val / 100)
            if annualized_to_fr(-10 / 100) <= val <= annualized_to_fr(15 / 100):
                self.theo_fr = val
            else:
                return "theo_fr must be between -10% and 15%"
        elif key == "bid_offset":
            if self.exit_offset <= val <= 1000:
                self.bid_offset = val
            else:
                return "bid_offset must be between exit_offset and 1000"
        elif key == "ask_offset":
            if self.exit_offset <= val <= 1000:
                self.ask_offset = val
            else:
                return "ask_offset must be between exit_offset and 1000"
        elif key == "exit_offset":
            if val <= min(self.bid_offset, self.ask_offset):
                self.exit_offset = val
            else:
                return "exit_offset must be lower than bid_offset and ask_offset"
        logging.info(f"Setting {key}: {val}")


def round_to_tick(value):
    return PRICE_TICK * round(value / PRICE_TICK)


def round_size(size):
    return SIZE_TICK * round(size / SIZE_TICK)


class Quotes:
    def __init__(self, cfg: Config, position, mark, index):
        self.theo = fr_to_price(cfg.theo_fr, index)
        self.bid_price = round_to_tick(self.theo - cfg.bid_offset)
        if self.bid_price < mark:
            self.bid_amt = round_size(max(min(cfg.size, cfg.max_pos - position), 0))
        else:
            self.bid_amt = 0
        self.ask_price = round_to_tick(self.theo + cfg.ask_offset)
        if self.ask_price > mark:
            self.ask_amt = round_size(max(min(cfg.size, -cfg.min_pos + position), 0))
        else:
            self.ask_amt = 0

        self.r_amt = round_size(cfg.tgt_pos - position)
        if self.r_amt > 0:
            self.r_side = Direction.BUY
            self.r_price = round_to_tick(self.theo - cfg.exit_offset)
        else:
            self.r_side = Direction.SELL
            self.r_price = round_to_tick(self.theo + cfg.exit_offset)


class Quoter:
    def __init__(self, tlx: thalex.Thalex, cfg: Config):
        self.tlx = tlx
        self.index: Optional[float] = None
        self.quotes: dict[thalex.Direction, Optional[dict]] = {
            Direction.BUY: {},
            Direction.SELL: {},
        }
        self.exit_quote: dict = {}
        self.position: Optional[float] = None
        self.cfg: Config = cfg
        self.tob = [None, None]
        self.fr = None
        self.mark = None

    async def adjust_order(
        self, side: Direction, price: float, amount: float, confirmed, quote_id: int
    ):
        assert confirmed is not None
        is_open = (confirmed.get("status") or "") in [
            "open",
            "partially_filled",
        ]
        if is_open:
            # if you have an exit quote and your position flips, you need to reinsert to change order direction
            if (
                Direction(confirmed["direction"]) != side
            ):  
                await self.tlx.cancel(client_order_id=quote_id, id=quote_id)
            elif (
                amount == 0
                or abs(confirmed["price"] - price) > self.cfg.amend_threshold
            ):
                logging.info(f"Amending {side} to {amount} @ {price}")
                await self.tlx.amend(
                    amount=amount,
                    price=price,
                    client_order_id=quote_id,
                    id=quote_id,
                )
        elif amount > 0:
            logging.info(f"Inserting {side}: {amount} @ {price}")
            await self.tlx.insert(
                amount=amount,
                price=price,
                direction=side,
                instrument_name=INSTRUMENT,
                client_order_id=quote_id,
                id=quote_id,
                label=ORDER_LABEL,
            )
            confirmed["status"] = "open"
            confirmed["price"] = price
            confirmed["direction"] = side

    async def update_quotes(self, new_index):
        up = self.index is None or new_index > self.index
        self.index = new_index
        if self.position is None or self.mark is None:
            return

        qs = Quotes(self.cfg, self.position, self.mark, self.index)
        assert qs.bid_price < qs.r_price < qs.ask_price
        adjustments = [
            self.adjust_order(
                side=Direction.SELL,
                price=qs.ask_price,
                amount=qs.ask_amt,
                confirmed=self.quotes[Direction.SELL],
                quote_id=QUOTE_ID[Direction.SELL],
            ),
            self.adjust_order(
                side=qs.r_side,
                price=qs.r_price,
                amount=abs(qs.r_amt),
                confirmed=self.exit_quote,
                quote_id=EXIT_QUOTE_ID,
            ),
            self.adjust_order(
                side=Direction.BUY,
                price=qs.bid_price,
                amount=qs.bid_amt,
                confirmed=self.quotes[Direction.BUY],
                quote_id=QUOTE_ID[Direction.BUY],
            ),
        ]
        if not up:
            adjustments.reverse()
        for a in adjustments:
            await a

    async def handle_notification(self, channel: str, notification):
        logging.debug(f"notification in channel {channel} {notification}")
        if channel == "session.orders":
            for order in notification:
                if order["client_order_id"] == EXIT_QUOTE_ID:
                    self.exit_quote = order
                else:
                    self.quotes[Direction(order["direction"])] = order
        elif channel.startswith("price_index"):
            await self.update_quotes(new_index=notification["price"])
        elif channel == "account.portfolio":
            await self.tlx.cancel_session()  # Cancel all orders in this session
            self.quotes = {Direction.BUY: {}, Direction.SELL: {}}
            try:
                self.position = next(
                    p for p in notification if p["instrument_name"] == INSTRUMENT
                )["position"]
            except StopIteration:
                self.position = self.position or 0
            logging.info(f"Portfolio updated - {INSTRUMENT} position: {self.position}")
        elif channel.startswith("ticker"):
            self.tob = [
                notification.get("best_bid_price"),
                notification.get("best_ask_price"),
            ]
            self.mark = notification.get("mark_price")
            self.fr = notification.get("funding_rate")

    async def quote(self):
        await self.tlx.connect()
        await self.tlx.login(
            keys.key_ids[keys.NETWORK], keys.private_keys[keys.NETWORK]
        )
        await self.tlx.set_cancel_on_disconnect(timeout_secs=6)
        await self.tlx.public_subscribe(
            [f"price_index.{INDEX}", f"ticker.{INSTRUMENT}.500ms"]
        )
        await self.tlx.private_subscribe(["session.orders", "account.portfolio"])
        while True:
            msg = await self.tlx.receive()
            msg = json.loads(msg)
            if "channel_name" in msg:
                await self.handle_notification(msg["channel_name"], msg["notification"])
            elif "result" in msg:
                logging.debug(msg)
            else:
                logging.error(msg)
                await self.tlx.cancel_session()
                self.quotes = {Direction.BUY: {}, Direction.SELL: {}}

    def get_cfg(self):
        return self.cfg

    def price_to_ann(self, price: Optional[float]):
        if self.index is None or self.index < 10:
            return None
        p = (price / self.index) - 1
        if p < -0.00025:
            return fr_to_annualized(p + 0.00025) * 100
        if p < 0.00025:
            return 0
        return fr_to_annualized(p - 0.00025) * 100

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("log.txt", mode="a"),
        ],
    )
    run = True  # We set this to false when we want to stop
    cfg = Config()
    while run:
        tlx = thalex.Thalex(network=keys.NETWORK)
        quoter = Quoter(tlx, cfg)
        task = asyncio.create_task(quoter.quote())
        try:
            await task
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Lost connection ({e}). Reconnecting...")
            time.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("Quoting cancelled")
            run = False
        except:
            logging.exception("There was an unexpected error:")
            run = False
        if tlx.connected():
            await tlx.cancel_session()
            await tlx.disconnect()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())



