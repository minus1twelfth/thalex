import enum
from typing import Optional


CID_IGNORE = -1
CID_INSTRUMENTS = 1
CID_SUBSCRIBE = 2
CID_QUOTE = 3
CID_CANCEL_SESSION = 4
CID_LOGIN = 5
CID_CANCEL_DISCO = 6
CID_MMP = 7


class InstrumentType(enum.Enum):
    PUT = "put"
    CALL = "call"
    FUT = "future"
    PERP = "perpetual"
    COMBO = "combo"


class Instrument:
    def __init__(self, name: str, expiry: float, exp_str: str, itype: InstrumentType, k: Optional[float]):
        self.name: str = name
        self.exp_str: str = exp_str
        self.exp: float = expiry
        self.type: InstrumentType = itype
        self.k: float = k


class Ticker:
    def __init__(self, delta, mark_iv, bid_iv, ask_iv):
        self.bid_iv: float = bid_iv
        self.ask_iv: float = ask_iv
        self.mark_iv: float = mark_iv
        self.delta: float = delta

    def __repr__(self):
        return f"d: {self.delta}, m:{self.mark_iv} b: {self.bid_iv}, a: {self.ask_iv}"


# [exp_str][k][itype] -> Ticker
IvStore = dict[str, dict[float, dict[InstrumentType, Ticker]]]