import dataclasses
import json
import logging
from dataclasses import asdict


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

	def persist(self):
		with open('settings.json', 'w') as file:
			json.dump(asdict(self), file, indent=2)

	@staticmethod
	def load():
		try:
			with open('settings.json', 'r') as file:
				data = json.load(file)
				return Settings(**data)
		except Exception as e:
			logging.warning(f'Failed to load settings because {repr(e)}')
			return default_settings()

	def enable_expiry(self, expiry: str, enabled: bool):
		if enabled:
			if expiry not in self.enabled_expiries:
				self.enabled_expiries.append(expiry)
		else:
			self.enabled_expiries.remove(expiry)

	def expiry_is_enabled(self, expiry: str):
		return expiry in self.enabled_expiries


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
