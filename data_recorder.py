from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RecordedSample:
  sample_id: int
  value: Any
  cycle: int


class DataRecorder:
  def __init__(self, name: str, capacity: int):
    self.name = name
    self.capacity = capacity
    self.samples: deque[RecordedSample] = deque(maxlen=capacity)

  def record(self, sample_id: int, value: Any, cycle: int) -> None:
    self.samples.append(RecordedSample(sample_id=sample_id, value=value, cycle=cycle))

  def record_token(self, token: Any, cycle: int) -> None:
    self.record(token.sample_id, token.value, cycle)

  def get_signal(self) -> list[Any]:
    return [sample.value for sample in self.samples]

  def get_cycles(self) -> list[int]:
    return [sample.cycle for sample in self.samples]

  def get_sample_ids(self) -> list[int]:
    return [sample.sample_id for sample in self.samples]
