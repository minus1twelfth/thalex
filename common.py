import datetime
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


def neighbours(chain: list[float], tgt_k: float) -> tuple[float, float]:
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


class InstrumentType(enum.Enum):
    PUT = "put"
    CALL = "call"
    FUT = "future"
    PERP = "perpetual"
    COMBO = "combo"

class Expiry:
    def __init__(self, ts):
        self.date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        self.name = self.date.strftime('%d%b%y').upper()

    def __lt__(self, other):
        return self.date < other.date

    def __eq__(self, other):
        return self.date == other.date

    def __hash__(self):
        return hash(self.date)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class Instrument:
    def __init__(self, name: str, expiry: Expiry, itype: InstrumentType, k: Optional[float]):
        self.name: str = name
        self.expiry: Expiry = expiry
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
IvStore = dict[Expiry, dict[float, dict[InstrumentType, Ticker]]]