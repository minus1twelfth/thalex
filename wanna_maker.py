import asyncio
import json
import logging
import socket
import time

import thalex as th
import websockets

from common import *
from deribit import Deribit
from gui import Gui
from quoter import Quoter
from settings import Settings

DERIBIT_URL = "wss://www.deribit.com/ws/api/v2"
NETWORK = th.Network.TEST
UNDERLYING = "BTC"
PRODUCT = "OBTCUSD"
assert UNDERLYING in PRODUCT


async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", )
    run = True  # We set this to false when we want to stop
    cfg = Settings.load()
    while run:
        iv_store = {}
        d = Deribit(iv_store, DERIBIT_URL, UNDERLYING)
        thalex = th.Thalex(network=NETWORK)
        q = Quoter(iv_store, thalex, cfg, NETWORK, UNDERLYING, PRODUCT)
        gui = await Gui.create(q)
        tasks = [
            asyncio.create_task(d.task()),
            asyncio.create_task(q.read_task()),
            asyncio.create_task(q.send_task()),
            asyncio.create_task(q.update_task()),
            asyncio.create_task(gui.run_websocket_server()),
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
        await gui.stop()
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
