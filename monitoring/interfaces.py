from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BatterySnapshot:
    level_percent: float
    charging: bool


@dataclass(slots=True)
class NetworkSnapshot:
    tx_bytes: int
    rx_bytes: int


class MockBatteryMonitor:
    def snapshot(self) -> BatterySnapshot:
        return BatterySnapshot(level_percent=100.0, charging=True)


class MockNetworkMonitor:
    def snapshot(self) -> NetworkSnapshot:
        return NetworkSnapshot(tx_bytes=0, rx_bytes=0)
