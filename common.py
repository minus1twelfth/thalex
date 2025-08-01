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
    COMBO = "combination"


class Expiry:
    def __init__(self, ts: Optional[float], name: Optional[str] = None):
        if ts is None:
            self.date = datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc)
            self.name = "PERP"
        else:
            self.date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            self.name = name or self.date.strftime('%d%b%y').upper()

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
    def __init__(self, name: str, expiry: Expiry, itype: InstrumentType, k: Optional[float], underlying:str = "unknown"):
        self.name: str = name
        self.expiry: Expiry = expiry
        self.type: InstrumentType = itype
        self.k: float = k
        self.underlying = underlying


def tlx_instrument(tlx_resp: dict) -> Instrument:
    itype = tlx_resp["type"]
    return Instrument(
        name=tlx_resp["instrument_name"],
        expiry=Expiry(tlx_resp.get("expiration_timestamp"), tlx_resp.get("expiry_date")),
        itype=InstrumentType(tlx_resp["option_type"] if itype == "option" else itype),
        k=tlx_resp.get("strike_price") or 0,
        underlying=tlx_resp["underlying"]
    )


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