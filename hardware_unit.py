from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from data_recorder import DataRecorder

class HardwareUnit(ABC):
  def __init__(self, name: str, latency_cycles: int = 1, is_fixed_point: bool = False):
    self.name: str = name
    self.latency_cycles: int = latency_cycles
    self.is_fixed_point: bool = is_fixed_point

    self.input_data: list = [] # data received from predecessor unit
    self.output_data: list = [] # computed result waiting to be forwarded

    self.busy: bool = False
    self.cycles_remaining: int = 0
    self.stalled_cycles: int = 0

    self.next_unit: HardwareUnit | None = None
    self.recorder: DataRecorder | None = None

  # Wires this unit's output to next_unit
  def connect(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit = next_unit
    return next_unit

  def tick(self, current_cycle: int) -> None:
    if self.output_data:
      self.handoff_to_next()

    if self.output_data:
      self.stalled_cycles += 1

    if not self.busy and self.input_data and not self.output_data:
      self.busy = True
      self.cycles_remaining = self.latency_cycles

    if self.busy:
      self.cycles_remaining -= 1

      if self.cycles_remaining == 0:
        self.output_data = self.compute(self.input_data)
        self.input_data = []
        self.busy = False
        
        if self.recorder:
          self.recorder.record(self.output_data)

        self.handoff_to_next()

  def handoff_to_next(self) -> None:
    if self.next_unit is None:
      self.output_data = []
      return

    if self.next_unit.is_available():
      self.next_unit.input_data = self.output_data
      self.output_data = []

  def start_work(self) -> None:
    self.busy = True
    self.cycles_remaining = self.latency_cycles

  def attach_recorder(self, recorder: DataRecorder) -> None:
    self.recorder = recorder

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
    return f"<{self.__class__.__name__} name={self.name} busy={self.busy} latency_cycles={self.latency_cycles} cycles_remaining={self.cycles_remaining} stalled_cycles={self.stalled_cycles}>"