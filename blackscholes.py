import math

SQRT_2PI = math.sqrt(math.pi * 2)


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


def call_delta(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        norm_d1 = 0.5 + 0.5 * math.erf(d1 / math.sqrt(2))
        return norm_d1
    elif fwd > k:
        return 1.0
    else:
        return 0.0


def put_delta(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        norm_d1 = 0.5 + 0.5 * math.erf(d1 / math.sqrt(2))
        return norm_d1 - 1
    elif fwd < k:
        return -1.0
    else:
        return 0.0


def theta(fwd: float, k: float, sigma: float, maturity: float) -> float:
    sqtime = math.sqrt(maturity)
    voltime = sqtime * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        inc_norm_d1 = math.pow(math.e, d1 * d1 * -0.5) / SQRT_2PI
        return -fwd * inc_norm_d1 * sigma / (2.0 * sqtime * 365.25)
    else:
        return 0.0


def gamma(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        inc_norm_d1 = math.pow(math.e, d1 * d1 * -0.5) / SQRT_2PI
        return inc_norm_d1 / (fwd * voltime)
    else:
        return 0.0


def vega(fwd: float, k: float, sigma: float, maturity: float) -> float:
    voltime = math.sqrt(maturity) * sigma
    if voltime > 0.0:
        d1 = math.log(fwd / k) / voltime + 0.5 * voltime
        inc_norm_d1 = math.pow(math.e, d1 * d1 * -0.5) / SQRT_2PI
        return 0.01 * fwd * inc_norm_d1 * voltime / sigma
    else:
        return 0
