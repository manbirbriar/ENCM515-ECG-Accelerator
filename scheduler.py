from __future__ import annotations

from hardware_unit import HardwareUnit, SampleToken


class Scheduler(HardwareUnit):
  def __init__(self, name: str, lanes: list[HardwareUnit]):
    super().__init__(name, latency_cycles=1, initiation_interval=1)
    self.lanes = lanes
    self.dispatch_index = 0

  def can_accept(self, current_cycle: int) -> bool:
    return any(lane.can_accept(current_cycle) for lane in self.lanes)

  def accept(self, token: SampleToken, current_cycle: int) -> bool:
    lane_count = len(self.lanes)
    for offset in range(lane_count):
      lane = self.lanes[(self.dispatch_index + offset) % lane_count]
      if lane.can_accept(current_cycle):
        lane.accept(token, current_cycle)
        self.dispatch_index = (self.dispatch_index + offset + 1) % lane_count
        self.accepted_count += 1
        self.emitted_count += 1
        return True

    self.stalled_cycles += 1
    return False

  def tick(self, current_cycle: int) -> None:
    return

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    return token
