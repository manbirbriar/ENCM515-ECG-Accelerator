from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from data_recorder import DataRecorder

class HardwareUnit(ABC):
  def __init__(self, name: str, latency_cycles: int = 1, is_fixed_point: bool = False):
    self.name: str = name
    self.latency_cycles: int = latency_cycles
    self.is_fixed_point: bool = is_fixed_point

    self.input_data = None    # scalar sample received from predecessor unit
    self.output_data = None   # computed scalar result waiting to be forwarded

    self.busy: bool = False
    self.cycles_remaining: int = 0

    # Performance tracking
    self.busy_cycles: int = 0
    self.idle_cycles: int = 0
    self.stalled_cycles: int = 0

    self.next_unit: HardwareUnit | None = None
    self.recorder: DataRecorder | None = None

  # Wires this unit's output to next_unit
  def connect(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit = next_unit
    return next_unit

  def tick(self, current_cycle: int) -> None:
    # Try to hand off any pending output first
    # Handles the case where downstream became available this cycle
    if self.output_data is not None:
      self.handoff_to_next()

    # If output still pending after handoff attempt, we are stalled
    # Count and return — nothing else to do until downstream clears
    if self.output_data is not None:
      self.stalled_cycles += 1
      return

    # Start work if input is available and we are not already busy
    if not self.busy and self.input_data is not None:
      self.busy = True
      self.cycles_remaining = self.latency_cycles

    # Count idle cycles when there is nothing to do
    if not self.busy and self.input_data is None:
      self.idle_cycles += 1
      return

    # Tick down latency counter
    if self.busy:
      self.busy_cycles += 1
      self.cycles_remaining -= 1

      if self.cycles_remaining == 0:
        self.output_data = self.compute(self.input_data)
        self.input_data = None
        self.busy = False

        if self.recorder:
          self.recorder.record(self.output_data)

        self.handoff_to_next()

        # If handoff failed, downstream was busy at completion — count as stall
        if self.output_data is not None:
          self.stalled_cycles += 1

  def handoff_to_next(self) -> None:
    if self.next_unit is None:
      self.output_data = None
      return

    if self.next_unit.is_available():
      self.next_unit.input_data = self.output_data
      self.output_data = None

  def attach_recorder(self, recorder: DataRecorder) -> None:
    self.recorder = recorder

  @abstractmethod
  def compute(self, data: Any) -> Any:
    return

  # Unit is available when it has no pending input and is not busy
  def is_available(self) -> bool:
    return not self.busy and self.input_data is None

  # Unit is stalled when it has output waiting but downstream is not ready
  def is_stalled(self) -> bool:
    return not self.busy and self.output_data is not None

  @property
  def utilization(self) -> float:
    total = self.busy_cycles + self.idle_cycles + self.stalled_cycles
    return self.busy_cycles / total if total > 0 else 0.0

  def __repr__(self) -> str:
    return (
      f"<{self.__class__.__name__} name={self.name} "
      f"busy={self.busy} latency_cycles={self.latency_cycles} "
      f"busy_cycles={self.busy_cycles} idle_cycles={self.idle_cycles} "
      f"stalled_cycles={self.stalled_cycles} "
      f"utilization={self.utilization:.1%}>"
    )
