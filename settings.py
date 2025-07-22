import dataclasses

@dataclasses.dataclass
class Settings:
	min_delta: float
	max_delta: float
	quote_size: float
	mmp_size: float
	label: str
	enabled_expiries: list[str]
	width_bid_call: float
	width_ask_call: float
	width_bid_put: float
	width_ask_put: float
	vol_offsets: dict[str, list[float]]
