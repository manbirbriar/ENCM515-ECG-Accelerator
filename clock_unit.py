from __future__ import annotations
from hardware_unit import HardwareUnit

class ClockUnit:
  """
  Global clock. On each tick(), increments cycle counter and calls tick() on every registered unit.
  Units are ticked in reverse registration order (back-to-front) to prevent 1-cycle delays between connected units.
  """
  def __init__(self):
    self.cycle: int = 0
    self.units: list[HardwareUnit] = []
    self.stalled_cycles: int = 0

  def subscribe(self, unit: HardwareUnit) -> "ClockUnit":
    self.units.append(unit)
    return self

  def subscribe_many(self, units: list[HardwareUnit]) -> "ClockUnit":
    for unit in units:
      self.subscribe(unit)
    return self

  def tick(self) -> int:
    self.cycle += 1

    for unit in reversed(self.units):
      unit.tick(self.cycle)

    if any(unit.is_stalled() for unit in self.units):
      self.stalled_cycles += 1

    return self.cycle

  def __repr__(self) -> str:
    return f"<ClockUnit cycle={self.cycle} units={len(self.units)} stalled_cycles={self.stalled_cycles}>"
