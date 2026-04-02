from __future__ import annotations

from hardware_unit import HardwareUnit


class ClockUnit:
  def __init__(self):
    self.cycle = 0
    self.units: list[HardwareUnit] = []
    self.stalled_cycles = 0

  def subscribe(self, unit: HardwareUnit) -> "ClockUnit":
    self.units.append(unit)
    return self

  def subscribe_many(self, units: list[HardwareUnit]) -> "ClockUnit":
    for unit in units:
      self.subscribe(unit)
    return self

  def tick(self) -> int:
    self.cycle += 1

    # Tick downstream first so samples can move across multiple stages in one cycle
    # when ready/valid conditions allow it.
    for unit in reversed(self.units):
      unit.tick(self.cycle)

    if any(unit.is_stalled() for unit in self.units):
      self.stalled_cycles += 1

    return self.cycle
