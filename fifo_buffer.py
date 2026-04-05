from __future__ import annotations
from collections import deque
from hardware_unit import HardwareUnit

class FIFOBuffer(HardwareUnit):
  """
  Between DataUploader and MACUnit.
  Accumulates samples while the MAC is busy.
  Models the input buffer that sits between the ADC and the DSP core in a real hardware system.
  """
  def __init__(self, name: str, capacity: int):
    super().__init__(name, latency_cycles=1)

    self.capacity = capacity
    self.queue: deque = deque()

    self.dropped_samples: int = 0
    self.max_depth: int = 0

  # Called by DataUploader to push one sample into the FIFO
  def push(self, sample) -> bool:
    if len(self.queue) >= self.capacity:
      self.dropped_samples += 1
      return False

    self.queue.append(sample)
    self.max_depth = max(self.max_depth, len(self.queue))
    return True

  # Called by MACUnit to pull one sample from the FIFO
  def pop(self):
    if self.queue:
      return self.queue.popleft()
    return None

  def is_empty(self) -> bool:
    return len(self.queue) == 0

  def has_data(self) -> bool:
    return len(self.queue) > 0

  # Always available to receive samples
  # Overflow is handled by dropped_samples
  def is_available(self) -> bool:
    return len(self.queue) < self.capacity

  def tick(self, current_cycle: int) -> None:
    # Track idle cycles when FIFO is empty
    if self.is_empty():
      self.idle_cycles += 1

  # Unused
  def compute(self, data):
    return data

  def __repr__(self) -> str:
    return f"<InputFIFO name={self.name} depth={len(self.queue)}/{self.capacity} max_depth={self.max_depth} dropped={self.dropped_samples}>"