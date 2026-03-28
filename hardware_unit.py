from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from data_recorder import DataRecorder

class HardwareUnit(ABC):
  def __init__(self, name: str, latency_cycles: int = 1):
    self.name: str = name
    self.latency_cycles: int = latency_cycles

    self.input_data: list = [] # data received from predecessor unit
    self.output_data: list = [] # computed result waiting to be forwarded

    self.busy: bool = False
    self.cycles_remaining: int = 0

    self.next_unit: HardwareUnit | None = None
    self.recorder: DataRecorder | None = None

  # Wires this unit's output to next_unit
  def connect(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit = next_unit
    return next_unit

  # Used to ensure that each unit only does 1 clock cycle of work per cycle
  def tick(self, current_cycle: int) -> None:
    self.current_cycle = current_cycle

    if not self.busy and self.input_data:
      self.start_work()

    if self.busy:
      self.cycles_remaining -= 1

      if self.cycles_remaining == 0:
        self.output_data = self.compute(self.input_data)
        self.input_data = []
        self.busy = False
        self.push_output()

    elif self.output_data:
      self.push_output()

  def start_work(self) -> None:
    self.busy = True
    self.cycles_remaining = self.latency_cycles

  def attach_recorder(self, recorder: DataRecorder) -> None:
    self.recorder = recorder

  def push_output(self) -> None:
    if self.recorder and self.output_data:
      self.recorder.record(self.output_data)

    if self.next_unit is None:
      return

    if not self.next_unit.busy and not self.next_unit.input_data:
      self.next_unit.input_data = self.output_data
      self.output_data = []

  @abstractmethod
  def compute(self, data: Any) -> Any:
    return

  # Determines if the hardware unit is available for computation
  def is_available(self) -> bool:
    return not self.busy and not self.input_data
  
  # A hardware unit is stalling if it has output data and is not busy
  def is_stalled(self) -> bool:
    return not self.busy and self.output_data

  def __repr__(self) -> str:
    return f"<{self.__class__.__name__} name={self.name} busy={self.busy} cycles_remaining={self.cycles_remaining}>"