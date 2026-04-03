from __future__ import annotations

from typing import Generic, TypeVar


T = TypeVar("T")


class CircularDelayLine(Generic[T]):
  def __init__(self, capacity: int, zero_value: T):
    self.capacity = capacity
    self.zero_value = zero_value
    self.storage: list[T] = [zero_value for _ in range(capacity)]
    self.write_index = 0
    self.count = 0

  def append(self, value: T) -> None:
    self.storage[self.write_index] = value
    self.write_index = (self.write_index + 1) % self.capacity
    self.count = min(self.count + 1, self.capacity)

  def delay(self, delay_samples: int) -> T:
    if delay_samples <= 0:
      raise ValueError("delay_samples must be positive")

    if self.count < delay_samples:
      return self.zero_value

    read_index = (self.write_index - delay_samples) % self.capacity
    return self.storage[read_index]


class CircularFIFO(Generic[T]):
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.storage: list[T | None] = [None for _ in range(capacity)]
    self.read_index = 0
    self.write_index = 0
    self.count = 0

  def push(self, value: T) -> bool:
    if self.count >= self.capacity:
      return False

    self.storage[self.write_index] = value
    self.write_index = (self.write_index + 1) % self.capacity
    self.count += 1
    return True

  def pop(self) -> T | None:
    if self.count == 0:
      return None

    value = self.storage[self.read_index]
    self.storage[self.read_index] = None
    self.read_index = (self.read_index + 1) % self.capacity
    self.count -= 1
    return value

  def __len__(self) -> int:
    return self.count

  def is_empty(self) -> bool:
    return self.count == 0

  def is_full(self) -> bool:
    return self.count >= self.capacity
