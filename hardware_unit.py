from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from data_recorder import DataRecorder

if TYPE_CHECKING:
  from clock_unit import ClockUnit

# Event names
INPUT_READY  = 'INPUT_READY'
COMPUTE_DONE = 'COMPUTE_DONE'

class HardwareUnit(ABC):
  def __init__(self, name: str, latency_cycles: int = 1, is_fixed_point: bool = False):
    self.name: str = name
    self.latency_cycles: int = latency_cycles
    self.is_fixed_point: bool = is_fixed_point

    self.input_data = None
    self.output_data = None

    self.busy: bool = False

    # Performance tracking
    self.busy_cycles: int = 0
    self.idle_cycles: int = 0
    self.stalled_cycles: int = 0
    self._last_active_cycle: int = 0  # used to calculate idle cycles
    self._stall_start_cycle: int = 0  # used to calculate stall cycles

    self.next_unit: HardwareUnit | None = None
    self._waiting_upstream: HardwareUnit | None = None  # unit stalled waiting for us
    self.recorder: DataRecorder | None = None

  def connect(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit = next_unit
    return next_unit

  def handle_event(self, event: str, cycle: int, clock: 'ClockUnit') -> None:
    if event == INPUT_READY:
      # Track idle time since last active cycle
      self.idle_cycles += cycle - self._last_active_cycle
      self._last_active_cycle = cycle

      self.busy = True
      self.input_data = self.input_data  # already set by upstream handoff
      clock.schedule(cycle + self.latency_cycles, self, COMPUTE_DONE)

    elif event == COMPUTE_DONE:
      self.output_data = self.compute(self.input_data)
      self.input_data = None
      self.busy = False
      self.busy_cycles += self.latency_cycles

      if self.recorder:
        self.recorder.record(self.output_data)

      self._try_handoff(cycle, clock)

  def _try_handoff(self, cycle: int, clock: 'ClockUnit') -> None:
    if self.next_unit is None:
      self.output_data = None
      # Notify any upstream unit that was stalled waiting for us
      self._notify_upstream(cycle, clock)
      return

    if self.next_unit.is_available():
      self.next_unit.input_data = self.output_data
      self.output_data = None
      self._last_active_cycle = cycle
      # Schedule downstream to start computing
      clock.schedule(cycle, self.next_unit, INPUT_READY)
      # Notify any upstream unit that was stalled waiting for us
      self._notify_upstream(cycle, clock)
    else:
      # Downstream busy — register as stalled
      self._stall_start_cycle = cycle
      self.next_unit._waiting_upstream = self

  def _notify_upstream(self, cycle: int, clock: 'ClockUnit') -> None:
    """Called when this unit completes and frees up space for upstream."""
    if self._waiting_upstream is not None:
      upstream = self._waiting_upstream
      self._waiting_upstream = None
      # Count stall cycles for upstream
      upstream.stalled_cycles += cycle - upstream._stall_start_cycle
      # Complete the handoff that was previously blocked
      self.input_data = upstream.output_data
      upstream.output_data = None
      clock.schedule(cycle, self, INPUT_READY)
      upstream._last_active_cycle = cycle

  def attach_recorder(self, recorder: DataRecorder) -> None:
    self.recorder = recorder

  @abstractmethod
  def compute(self, data: Any) -> Any:
    return

  def is_available(self) -> bool:
    return not self.busy and self.input_data is None

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
