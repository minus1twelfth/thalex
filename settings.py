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


def default_settings() -> Settings:
	return Settings(
		min_delta=0.2,
		max_delta=0.8,
		quote_size=0.1,
		mmp_size=0.3,
		label="wannabot",
		width_bid_call=50,
		width_ask_call=100,
		width_bid_put=50,
		width_ask_put=100,
		vol_offsets={},
		enabled_expiries=[]
	)
