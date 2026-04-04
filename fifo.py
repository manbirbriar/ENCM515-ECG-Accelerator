from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from clock_unit import ClockUnit

class InputFIFO:
  """
  FIFO buffer between DataUploader and MACUnit.
  Not a HardwareUnit — it is a pure storage structure.
  DataUploader pushes into it, MACUnit pulls from it.
  """
  def __init__(self, name: str, capacity: int):
    self.name = name
    self.capacity = capacity
    self.queue: deque = deque()

    self.dropped_samples: int = 0
    self.max_depth: int = 0

  def push(self, sample) -> bool:
    if len(self.queue) >= self.capacity:
      self.dropped_samples += 1
      return False
    self.queue.append(sample)
    self.max_depth = max(self.max_depth, len(self.queue))
    return True

  def pop(self):
    if self.queue:
      return self.queue.popleft()
    return None

  def has_data(self) -> bool:
    return len(self.queue) > 0

  def is_empty(self) -> bool:
    return len(self.queue) == 0

  def is_available(self) -> bool:
    return len(self.queue) < self.capacity

  def __repr__(self) -> str:
    return (
      f"<InputFIFO name={self.name} "
      f"depth={len(self.queue)}/{self.capacity} "
      f"max_depth={self.max_depth} "
      f"dropped={self.dropped_samples}>"
    )
