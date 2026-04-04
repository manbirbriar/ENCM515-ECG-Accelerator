from __future__ import annotations
from circular_buffer import CircularBuffer
from fifo import InputFIFO
from data_recorder import DataRecorder
from typing import TYPE_CHECKING
from config import (
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_SHIFT_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES,
  FIXED_POINT_BITS
)

if TYPE_CHECKING:
  from clock_unit import ClockUnit

# Event names
SAMPLE_READY = 'SAMPLE_READY'
LP_DONE      = 'LP_DONE'
HP_DONE      = 'HP_DONE'
DV_DONE      = 'DV_DONE'

class MACUnit:
  """
  Time-multiplexed MAC unit running three FIR/IIR filter kernels sequentially:
    1. Low-pass:   y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
    2. High-pass:  y[n] = y[n-1] - (1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
    3. Derivative: y[n] = (1/8)[2x[n] + x[n-1] - x[n-3] - 2x[n-4]]

  Shared hardware: one multiplier, one accumulator, coefficient ROM, program sequencer.
  Each kernel reads from its own history buffer.
  """
  def __init__(self, name: str, fifo: InputFIFO, is_fixed_point: bool):
    self.name = name
    self.fifo = fifo
    self.is_fixed_point = is_fixed_point

    # Per-kernel latencies
    if is_fixed_point:
      self.lp_cycles = (FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + FIXED_ADD_CYCLES) * 2
      self.hp_cycles = (2*FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + 2*FIXED_ADD_CYCLES) + FIXED_ADD_CYCLES
      self.dv_cycles = 2*FIXED_SHIFT_CYCLES + FIXED_ADD_CYCLES + 2*FIXED_SUB_CYCLES
    else:
      self.lp_cycles = (FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES) * 2
      self.hp_cycles = (2*FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + 2*FLOAT_ADD_CYCLES) + FLOAT_ADD_CYCLES
      self.dv_cycles = 2*FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES + 2*FLOAT_SUB_CYCLES

    self.latency_cycles = self.lp_cycles + self.hp_cycles + self.dv_cycles

    # History buffers
    dtype = int if is_fixed_point else float
    self.input_buffer = CircularBuffer(13, dtype=dtype)  # LP: x[n]..x[n-12]
    self.lp_buffer    = CircularBuffer(33, dtype=dtype)  # HP: x[n]..x[n-32]
    self.hp_buffer    = CircularBuffer(5,  dtype=dtype)  # DV: x[n]..x[n-4]

    # IIR state registers
    self.lp_y1 = 0
    self.lp_y2 = 0
    self.hp_y1 = 0

    # Intermediate results
    self._current_sample = None
    self._lp_result = None
    self._hp_result = None

    # State
    self._idle = True
    self.output_data = None
    self.next_unit = None
    self._waiting_upstream = None
    self._stall_start_cycle = 0
    self._last_active_cycle = 0

    # Performance tracking
    self.busy_cycles: int = 0
    self.idle_cycles: int = 0
    self.stalled_cycles: int = 0

    # Per-stage recorders
    self.raw_recorder: DataRecorder | None = None
    self.lp_recorder:  DataRecorder | None = None
    self.hp_recorder:  DataRecorder | None = None
    self.dv_recorder:  DataRecorder | None = None

  def connect(self, next_unit) -> any:
    self.next_unit = next_unit
    return next_unit

  def attach_raw_recorder(self, r): self.raw_recorder = r
  def attach_lp_recorder(self, r):  self.lp_recorder  = r
  def attach_hp_recorder(self, r):  self.hp_recorder  = r
  def attach_dv_recorder(self, r):  self.dv_recorder  = r

  def is_idle(self) -> bool:
    return self._idle and self.output_data is None

  def is_available(self) -> bool:
    return self.is_idle()

  def is_stalled(self) -> bool:
    return not self._idle and self.output_data is not None

  # --- Kernel implementations ---

  def _run_lp_kernel(self, sample):
    self.input_buffer.push(sample)
    x_n   = self.input_buffer[-1]
    x_n6  = self.input_buffer[-7]
    x_n12 = self.input_buffer[-13]
    if self.is_fixed_point:
      part_a = x_n - (int(x_n6) << 1) + x_n12
      y_n = (int(self.lp_y1) << 1) - int(self.lp_y2) + int(part_a)
    else:
      part_a = x_n - 2*x_n6 + x_n12
      y_n = 2*self.lp_y1 - self.lp_y2 + part_a
    self.lp_y2 = self.lp_y1
    self.lp_y1 = y_n
    if self.lp_recorder: self.lp_recorder.record(y_n)
    return y_n

  def _run_hp_kernel(self, lp_sample):
    self.lp_buffer.push(lp_sample)
    x_n   = self.lp_buffer[-1]
    x_n16 = self.lp_buffer[-17]
    x_n17 = self.lp_buffer[-18]
    x_n32 = self.lp_buffer[-33]
    if self.is_fixed_point:
      part_a = -(int(x_n) >> 5) + int(x_n16) - int(x_n17) + (int(x_n32) >> 5)
    else:
      part_a = -(x_n/32) + x_n16 - x_n17 + (x_n32/32)
    y_n = self.hp_y1 + part_a
    self.hp_y1 = y_n
    if self.hp_recorder: self.hp_recorder.record(y_n)
    return y_n

  def _run_dv_kernel(self, hp_sample):
    self.hp_buffer.push(hp_sample)
    x_n  = self.hp_buffer[-1]
    x_n1 = self.hp_buffer[-2]
    x_n3 = self.hp_buffer[-4]
    x_n4 = self.hp_buffer[-5]
    if self.is_fixed_point:
      y_n = int(2*int(x_n) + int(x_n1) - int(x_n3) - 2*int(x_n4)) >> 3
    else:
      y_n = (1/8) * (2*x_n + x_n1 - x_n3 - 2*x_n4)
    if self.dv_recorder: self.dv_recorder.record(y_n)
    return y_n

  # --- Event handler ---

  def handle_event(self, event: str, cycle: int, clock: 'ClockUnit') -> None:
    if event == SAMPLE_READY:
      if not self.fifo.has_data():
        return

      # Track idle time
      self.idle_cycles += cycle - self._last_active_cycle

      sample = self.fifo.pop()
      if self.raw_recorder: self.raw_recorder.record(sample)

      self._current_sample = sample
      self._idle = False
      clock.schedule(cycle + self.lp_cycles, self, LP_DONE)
      self.busy_cycles += self.lp_cycles

    elif event == LP_DONE:
      self._lp_result = self._run_lp_kernel(self._current_sample)
      clock.schedule(cycle + self.hp_cycles, self, HP_DONE)
      self.busy_cycles += self.hp_cycles

    elif event == HP_DONE:
      self._hp_result = self._run_hp_kernel(self._lp_result)
      clock.schedule(cycle + self.dv_cycles, self, DV_DONE)
      self.busy_cycles += self.dv_cycles

    elif event == DV_DONE:
      dv_result = self._run_dv_kernel(self._hp_result)
      self._idle = True
      self._last_active_cycle = cycle
      self.output_data = dv_result
      self._try_handoff(cycle, clock)

      # If FIFO has more data and we are now idle, process next sample
      if self._idle and self.fifo.has_data():
        clock.schedule(cycle, self, SAMPLE_READY)

  def _try_handoff(self, cycle: int, clock: 'ClockUnit') -> None:
    if self.next_unit is None:
      self.output_data = None
      self._notify_upstream(cycle, clock)
      return

    if self.next_unit.is_available():
      self.next_unit.input_data = self.output_data
      self.output_data = None
      clock.schedule(cycle, self.next_unit, 'INPUT_READY')
      self._notify_upstream(cycle, clock)
    else:
      self._stall_start_cycle = cycle
      self.next_unit._waiting_upstream = self

  def _notify_upstream(self, cycle: int, clock: 'ClockUnit') -> None:
    if self._waiting_upstream is not None:
      upstream = self._waiting_upstream
      self._waiting_upstream = None
      upstream.stalled_cycles += cycle - upstream._stall_start_cycle
      self.input_data = upstream.output_data
      upstream.output_data = None
      clock.schedule(cycle, self, 'INPUT_READY')
      upstream._last_active_cycle = cycle

  @property
  def utilization(self) -> float:
    total = self.busy_cycles + self.idle_cycles + self.stalled_cycles
    return self.busy_cycles / total if total > 0 else 0.0

  def __repr__(self) -> str:
    return (
      f"<MACUnit name={self.name} "
      f"idle={self._idle} "
      f"busy_cycles={self.busy_cycles} "
      f"idle_cycles={self.idle_cycles} "
      f"stalled_cycles={self.stalled_cycles} "
      f"utilization={self.utilization:.1%}>"
    )
