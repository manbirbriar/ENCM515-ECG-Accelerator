from __future__ import annotations

from circular_buffer import CircularFIFO
from hardware_unit import HardwareUnit, SampleToken


class InputBuffer(HardwareUnit):
  def __init__(self, name: str, queue_size: int):
    super().__init__(name, latency_cycles=1, initiation_interval=1)
    self.queue_size = queue_size
    self.queue = CircularFIFO[SampleToken](capacity=queue_size)
    self.overflow_count = 0
    self.max_occupancy = 0
    self.forwarded_count = 0

  def accept(self, token: SampleToken, current_cycle: int) -> bool:
    if self.queue.is_full():
      self.overflow_count += 1
      self.stalled_cycles += 1
      return False

    self.queue.push(token)
    self.max_occupancy = max(self.max_occupancy, len(self.queue))
    self.accepted_count += 1
    return True

  def can_accept(self, current_cycle: int) -> bool:
    return not self.queue.is_full()

  def tick(self, current_cycle: int) -> None:
    if self.queue.is_empty():
      self.idle_cycles += 1
      return

    self.busy_cycles += 1
    if self.next_unit is None:
      self.queue.pop()
      self.forwarded_count += 1
      return

    if self.next_unit.can_accept(current_cycle):
      token = self.queue.pop()
      if token is not None:
        self.next_unit.accept(token, current_cycle)
        self.forwarded_count += 1
        self.emitted_count += 1
        return

    self.downstream_stall_cycles += 1

  def occupancy(self) -> int:
    return len(self.queue)

  def is_empty(self) -> bool:
    return self.queue.is_empty()

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    return token


SampleQueue = InputBuffer
