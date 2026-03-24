from __future__ import annotations
from hardware_unit import HardwareUnit

# Every HardwareUnit must be registered here before the simulation runs
# On each call to tick(), the clock increments its cycle counter and calls tick() on every registered unit
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

    # Ticking in reverse order (back-to-front) prevents artificial 1-cycle delays between connected units
    # TODO: I'm not 100% sure on this, but I am convincing myself that it makes sense
    for unit in reversed(self.units):
      unit.tick(self.cycle)

    if any(unit.is_stalled() for unit in self.units):
      self.stalled_cycles += 1

    return self.cycle

  def __repr__(self) -> str:
    return f"<ClockUnit cycle={self.cycle} units={len(self.units)} stalled_cycles={self.stalled_cycles}"