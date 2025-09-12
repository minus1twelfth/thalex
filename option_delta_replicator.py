import asyncio
import csv
import json

from dataclasses import dataclass, asdict
from typing import Optional, Union

import keys
import thalex as th

network = th.Network.TEST
KEY_ID = keys.key_ids[network]
PRIV_KEY = keys.private_keys[network]

OPTION = "BTC-31MAY24-70000-C"
PERPETUAL = "BTC-PERPETUAL"
HEDGING_BAND = 0.01

PORTFOLIO_REQUEST_ID = 1001
ORDER_REQUEST_ID = 1002
PERPETUAL_REQUEST_ID = 1003
OPTION_REQUEST_ID = 1004


@dataclass
class TradingData:
    delta: Optional[float] = None
    position: Optional[float] = None
    direction: Optional[str] = None
    ordered: Optional[float] = None
    filled: Optional[float] = None
    mark_perp: Optional[float] = None
    mark_option: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class Error:
    api: Optional[str] = None
    exception: Optional[Exception] = None


async def run_bot():
    data = TradingData()
    thalex = th.Thalex(network=network)
    await thalex.connect()
    await thalex.login(KEY_ID, PRIV_KEY)
    await thalex.portfolio(PORTFOLIO_REQUEST_ID)
    await thalex.ticker(PERPETUAL, PERPETUAL_REQUEST_ID)
    await thalex.ticker(OPTION, OPTION_REQUEST_ID)

    while True:
        msg = json.loads(await thalex.receive())
        msg_id = msg.get("id")

        if "error" in msg:
            return Error(api=msg["error"]["message"])

        if msg_id == PORTFOLIO_REQUEST_ID:
            try:
                res = next(
                    position
                    for position in msg["result"]
                    if position["instrument_name"] == PERPETUAL
                )
                data.position = res["position"]
            except StopIteration:  # if no position msg['result'] = None
                data.position = 0

        elif msg_id == OPTION_REQUEST_ID:
            data.delta = round(msg["result"]["delta"], ndigits=4)
            data.mark_option = msg["result"]["mark_price"]

        elif msg_id == PERPETUAL_REQUEST_ID:
            data.mark_perp = msg["result"]["mark_price"]
            data.timestamp = msg["result"]["mark_timestamp"]

        elif msg_id == ORDER_REQUEST_ID:
            data.filled = msg["result"]["filled_amount"]
            data.direction = msg["result"]["direction"]
            print(f"{data.direction.upper()} order executed, filled {data.filled:.4f}")
            return data

        if data.delta is not None and data.position is not None:
            offset = round(data.delta - data.position, 3)
            if abs(offset) > HEDGING_BAND:
                print(f"Sending perpetual order for {offset}")
                await thalex.insert(
                    id=ORDER_REQUEST_ID,
                    direction=(th.Direction.BUY if offset > 0 else th.Direction.SELL),
                    instrument_name=PERPETUAL,
                    order_type=th.OrderType.MARKET,
                    amount=abs(offset),
                )
                data.ordered = offset
            else:
                print(
                    f"Delta of {data.delta} - {data.position} = {offset} does not exceed hedging band ({HEDGING_BAND}), no need to do anything"
                )
                return data


def to_csv(data: Union[TradingData, Error], filename: str):
    with open(filename, "a", newline="") as file:
        w = csv.writer(file)
        if file.tell() == 0:
            w.writerow(asdict(data).keys())
        w.writerow(asdict(data).values())
        print(f"Data written:\n{data}")


def run_bot_and_handle_exceptions():
    try:
        loop = asyncio.get_event_loop()
        task = loop.create_task(run_bot())
        return loop.run_until_complete(task)
    except Exception as e:
        print(f"Error: {e}")
        return Error(exception=str(e))


if __name__ == "__main__":
    data = run_bot_and_handle_exceptions()
    if isinstance(data, TradingData):
        to_csv(data, "results.csv")
    elif isinstance(data, Error):
        to_csv(data, "errors.csv")
    else:
        print("No data captured.")

