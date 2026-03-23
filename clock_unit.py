from __future__ import annotations
from hardware_unit import HardwareUnit

class ClockUnit:
  def __init__(self):
    self.cycle: int = 0
    self.units: list[HardwareUnit] = []
    self.stalled_cycles: int = 0

  # subscribes a hardware unit to the clock unit
  def subscribe(self, unit: HardwareUnit) -> ClockUnit:
    self.units.append(unit)
    return self

  # subscribes multiple hardware units to the clock unit
  def subscribe_many(self, units: list[HardwareUnit]) -> ClockUnit:
    for unit in units:
      self.subscribe(unit)
    return self

  # sends tick event to subscribed hardware units
  def tick(self) -> int:
    self.cycle += 1

    for unit in self.units:
      unit.tick(self.cycle)

    if any(unit.is_stalled() for unit in self.units):
      self.stalled_cycles += 1

    return self.cycle

  def __repr__(self) -> str:
    return f"<ClockUnit cycle={self.cycle} units={len(self.units)} stalled_cycles={self.stalled_cycles}"