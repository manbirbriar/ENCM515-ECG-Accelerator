from __future__ import annotations
from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer
from fifo import InputFIFO
from data_recorder import DataRecorder
from config import (
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_SHIFT_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES
)

# Internal state machine states
IDLE         = "IDLE"
LP_BUSY      = "LP_BUSY"
HP_BUSY      = "HP_BUSY"
DV_BUSY      = "DV_BUSY"

class MACUnit(HardwareUnit):
  """
  Time-multiplexed MAC unit that runs three FIR/IIR filter kernels sequentially:
    1. Low-pass filter:   y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
    2. High-pass filter:  y[n] = y[n-1] - (1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
    3. Derivative filter: y[n] = (1/8)[2x[n] + x[n-1] - x[n-3] - 2x[n-4]]

  A single multiplier and accumulator are shared across all three kernels.
  The program sequencer loads the appropriate coefficients and reads from the
  appropriate history buffer for each kernel.

  History buffers:
    input_buffer[33]  raw ECG history          (LP reads from here)
    lp_buffer[33]     LP output history         (HP reads from here)
    hp_buffer[5]      HP output history         (Derivative reads from here)

  IIR state registers:
    lp_y1, lp_y2      LowPass recurrence state
    hp_y1             HighPass recurrence state
  """
  def __init__(self, name: str, fifo: InputFIFO, is_fixed_point: bool):

    # Per-sample latency for each kernel
    if is_fixed_point:
      # LowPass: FIR (shift + sub + add) + IIR (shift + sub + add)
      self.lp_cycles = (FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + FIXED_ADD_CYCLES) * 2
      # HighPass: FIR (2*shift + sub + 2*add) + IIR (add)
      self.hp_cycles = (2*FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + 2*FIXED_ADD_CYCLES) + FIXED_ADD_CYCLES
      # Derivative: FIR only (2*shift + add + 2*sub)
      self.dv_cycles = 2*FIXED_SHIFT_CYCLES + FIXED_ADD_CYCLES + 2*FIXED_SUB_CYCLES
    else:
      # LowPass: FIR (mul + sub + add) + IIR (mul + sub + add)
      self.lp_cycles = (FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES) * 2
      # HighPass: FIR (2*mul + sub + 2*add) + IIR (add)
      self.hp_cycles = (2*FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + 2*FLOAT_ADD_CYCLES) + FLOAT_ADD_CYCLES
      # Derivative: FIR only (2*mul + add + 2*sub)
      self.dv_cycles = 2*FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES + 2*FLOAT_SUB_CYCLES

    total_latency = self.lp_cycles + self.hp_cycles + self.dv_cycles
    super().__init__(name, latency_cycles=total_latency, is_fixed_point=is_fixed_point)

    self.fifo = fifo

    # Internal history buffers (shift registers in hardware)
    self.input_buffer = CircularBuffer(13, dtype=int if is_fixed_point else float) # TODO: Used to be 33, changed to 13 (WHY?)
    self.lp_buffer    = CircularBuffer(33, dtype=int if is_fixed_point else float)
    self.hp_buffer    = CircularBuffer(5,  dtype=int if is_fixed_point else float)

    # IIR state registers
    self.lp_y1 = 0
    self.lp_y2 = 0
    self.hp_y1 = 0

    # Internal state machine
    self.state = IDLE
    self.kernel_cycles_remaining = 0

    # Intermediate kernel results
    self._current_sample = None
    self._lp_result = None
    self._hp_result = None

    # Optional per-stage recorders
    self.raw_recorder: DataRecorder | None = None
    self.lp_recorder:  DataRecorder | None = None
    self.hp_recorder:  DataRecorder | None = None
    self.dv_recorder:  DataRecorder | None = None

  def attach_raw_recorder(self, recorder: DataRecorder) -> None:
    self.raw_recorder = recorder

  def attach_lp_recorder(self, recorder: DataRecorder) -> None:
    self.lp_recorder = recorder

  def attach_hp_recorder(self, recorder: DataRecorder) -> None:
    self.hp_recorder = recorder

  def attach_dv_recorder(self, recorder: DataRecorder) -> None:
    self.dv_recorder = recorder

  # --- Kernel implementations ---

  def _run_lp_kernel(self, sample) -> int | float:
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

    if self.lp_recorder:
      self.lp_recorder.record(y_n)

    return y_n

  def _run_hp_kernel(self, lp_sample) -> int | float:
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

    if self.hp_recorder:
      self.hp_recorder.record(y_n)

    return y_n

  def _run_dv_kernel(self, hp_sample) -> int | float:
    self.hp_buffer.push(hp_sample)

    x_n  = self.hp_buffer[-1]
    x_n1 = self.hp_buffer[-2]
    x_n3 = self.hp_buffer[-4]
    x_n4 = self.hp_buffer[-5]

    if self.is_fixed_point:
      y_n = int(2*int(x_n) + int(x_n1) - int(x_n3) - 2*int(x_n4)) >> 3
    else:
      y_n = (1/8) * (2*x_n + x_n1 - x_n3 - 2*x_n4)

    if self.dv_recorder:
      self.dv_recorder.record(y_n)

    return y_n

  # --- State machine tick ---

  def tick(self, current_cycle: int) -> None:
    # Try to hand off any pending output first
    if self.output_data is not None:
      self.handoff_to_next()

    # If handoff failed, we are stalled — nothing else to do this cycle
    if self.output_data is not None:
      self.stalled_cycles += 1
      return

    if self.state == IDLE:
      if self.fifo.has_data():
        sample = self.fifo.pop()

        if self.raw_recorder:
          self.raw_recorder.record(sample)

        self._current_sample = sample
        self.state = LP_BUSY
        self.kernel_cycles_remaining = self.lp_cycles
      else:
        self.idle_cycles += 1

    elif self.state == LP_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        self._lp_result = self._run_lp_kernel(self._current_sample)
        self.state = HP_BUSY
        self.kernel_cycles_remaining = self.hp_cycles

    elif self.state == HP_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        self._hp_result = self._run_hp_kernel(self._lp_result)
        self.state = DV_BUSY
        self.kernel_cycles_remaining = self.dv_cycles

    elif self.state == DV_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        dv_result = self._run_dv_kernel(self._hp_result)
        self.output_data = dv_result
        self.state = IDLE

        self.handoff_to_next()

        # If handoff failed downstream was busy at completion — count as stall
        if self.output_data is not None:
          self.stalled_cycles += 1

  def is_available(self) -> bool:
    return self.state == IDLE and self.output_data is None

  def is_stalled(self) -> bool:
    return self.output_data is not None

  # compute() unused — MAC overrides tick() directly
  def compute(self, data):
    return data

  def __repr__(self) -> str:
    return (
      f"<MACUnit name={self.name} "
      f"state={self.state} "
      f"busy_cycles={self.busy_cycles} "
      f"idle_cycles={self.idle_cycles} "
      f"stalled_cycles={self.stalled_cycles} "
      f"utilization={self.utilization:.1%}>"
    )
