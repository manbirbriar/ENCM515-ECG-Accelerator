from __future__ import annotations
from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer
from fifo import InputFIFO
from data_recorder import DataRecorder
from config import (
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_SHIFT_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES,
  FIXED_POINT_BITS
)

# Internal state machine states
IDLE         = "IDLE"
LP_BUSY_L0   = "LP_BUSY_L0"
HP_BUSY_L0   = "HP_BUSY_L0"
DV_BUSY_L0   = "DV_BUSY_L0"
LP_BUSY_L1   = "LP_BUSY_L1"
HP_BUSY_L1   = "HP_BUSY_L1"
DV_BUSY_L1   = "DV_BUSY_L1"

class SharedMACUnit(HardwareUnit):
  """
  Shared time-multiplexed MAC unit that processes two ECG leads sequentially
  on a single hardware datapath.

  For each sample pair, the MAC processes:
    Lead 0: LP → HP → DV
    Lead 1: LP → HP → DV

  This models a single physical MAC with a program sequencer that alternates
  between leads, using separate history buffers and IIR state registers per lead
  but sharing the multiplier, adder, and barrel shifter.

  Stalls occur when:
    - Both FIFOs empty (idle — waiting for ADC)
    - One FIFO empty when starting a new sample pair (structural hazard)
    - Downstream unit busy when MAC tries to hand off (pipeline stall)

  Per-lead latency is identical to the single-lane MACUnit.
  Total latency per sample pair = 2 * (lp + hp + dv) cycles.

  Architecture tradeoff vs dual MACUnit:
    - Half the multiplier/adder hardware
    - Same history buffer area (buffers cannot be shared — different signal history)
    - 2x cycles per sample pair in fixed point (34 vs 17) — still fits in 555 cycle budget
    - Float mode cannot keep up: 2 * 537 = 1074 > 555 cycles — FIFO overflow guaranteed
  """
  def __init__(self, name: str, fifo0: InputFIFO, fifo1: InputFIFO, is_fixed_point: bool):

    # Per-kernel latency (same as single MACUnit)
    if is_fixed_point:
      self.lp_cycles = (FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + FIXED_ADD_CYCLES) * 2
      self.hp_cycles = (2*FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + 2*FIXED_ADD_CYCLES) + FIXED_ADD_CYCLES
      self.dv_cycles = 2*FIXED_SHIFT_CYCLES + FIXED_ADD_CYCLES + 2*FIXED_SUB_CYCLES
    else:
      self.lp_cycles = (FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES) * 2
      self.hp_cycles = (2*FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + 2*FLOAT_ADD_CYCLES) + FLOAT_ADD_CYCLES
      self.dv_cycles = 2*FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES + 2*FLOAT_SUB_CYCLES

    # Total latency = both leads processed sequentially
    total_latency = 2 * (self.lp_cycles + self.hp_cycles + self.dv_cycles)
    super().__init__(name, latency_cycles=total_latency, is_fixed_point=is_fixed_point)

    self.fifo0 = fifo0
    self.fifo1 = fifo1

    # --- Per-lead history buffers ---
    # These CANNOT be shared — each lead has different signal history
    dtype = int if is_fixed_point else float

    self.input_buffer_l0 = CircularBuffer(13, dtype=dtype)
    self.lp_buffer_l0    = CircularBuffer(33, dtype=dtype)
    self.hp_buffer_l0    = CircularBuffer(5,  dtype=dtype)

    self.input_buffer_l1 = CircularBuffer(13, dtype=dtype)
    self.lp_buffer_l1    = CircularBuffer(33, dtype=dtype)
    self.hp_buffer_l1    = CircularBuffer(5,  dtype=dtype)

    # --- Per-lead IIR state registers ---
    # These CANNOT be shared — each lead has different filter state
    self.lp_y1_l0 = 0
    self.lp_y2_l0 = 0
    self.hp_y1_l0 = 0

    self.lp_y1_l1 = 0
    self.lp_y2_l1 = 0
    self.hp_y1_l1 = 0

    # --- State machine ---
    self.state = IDLE
    self.kernel_cycles_remaining = 0

    # Intermediate results per lead
    self._current_sample_l0 = None
    self._current_sample_l1 = None
    self._lp_result_l0 = None
    self._lp_result_l1 = None
    self._hp_result_l0 = None
    self._hp_result_l1 = None

    # Two output slots — one per lead
    self.output_data_l0 = None
    self.output_data_l1 = None

    # Two downstream units — one per lead
    self.next_unit_l1: HardwareUnit | None = None

    # Per-lead recorders
    self.raw_recorder_l0: DataRecorder | None = None
    self.raw_recorder_l1: DataRecorder | None = None
    self.lp_recorder_l0:  DataRecorder | None = None
    self.lp_recorder_l1:  DataRecorder | None = None
    self.hp_recorder_l0:  DataRecorder | None = None
    self.hp_recorder_l1:  DataRecorder | None = None
    self.dv_recorder_l0:  DataRecorder | None = None
    self.dv_recorder_l1:  DataRecorder | None = None

    # Stall reason tracking
    self.fifo_stall_cycles: int = 0   # stalled waiting for both FIFOs to have data
    self.output_stall_cycles: int = 0 # stalled waiting for downstream to accept output

  # --- Recorder attachment ---

  def attach_recorders_l0(self, raw, lp, hp, dv):
    self.raw_recorder_l0 = raw
    self.lp_recorder_l0  = lp
    self.hp_recorder_l0  = hp
    self.dv_recorder_l0  = dv

  def attach_recorders_l1(self, raw, lp, hp, dv):
    self.raw_recorder_l1 = raw
    self.lp_recorder_l1  = lp
    self.hp_recorder_l1  = hp
    self.dv_recorder_l1  = dv

  # Connect lead 0 downstream (uses parent connect())
  # Connect lead 1 downstream
  def connect_l1(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit_l1 = next_unit
    return next_unit

  # --- Kernel implementations ---

  def _run_lp_kernel(self, sample, lead: int) -> int | float:
    buf = self.input_buffer_l0 if lead == 0 else self.input_buffer_l1
    lp_y1 = self.lp_y1_l0 if lead == 0 else self.lp_y1_l1
    lp_y2 = self.lp_y2_l0 if lead == 0 else self.lp_y2_l1
    rec = self.lp_recorder_l0 if lead == 0 else self.lp_recorder_l1

    buf.push(sample)
    x_n   = buf[-1]
    x_n6  = buf[-7]
    x_n12 = buf[-13]

    if self.is_fixed_point:
      part_a = x_n - (int(x_n6) << 1) + x_n12
      y_n = (int(lp_y1) << 1) - int(lp_y2) + int(part_a)
    else:
      part_a = x_n - 2*x_n6 + x_n12
      y_n = 2*lp_y1 - lp_y2 + part_a

    if lead == 0:
      self.lp_y2_l0 = self.lp_y1_l0
      self.lp_y1_l0 = y_n
    else:
      self.lp_y2_l1 = self.lp_y1_l1
      self.lp_y1_l1 = y_n

    if rec:
      rec.record(y_n)
    return y_n

  def _run_hp_kernel(self, lp_sample, lead: int) -> int | float:
    buf = self.lp_buffer_l0 if lead == 0 else self.lp_buffer_l1
    hp_y1 = self.hp_y1_l0 if lead == 0 else self.hp_y1_l1
    rec = self.hp_recorder_l0 if lead == 0 else self.hp_recorder_l1

    buf.push(lp_sample)
    x_n   = buf[-1]
    x_n16 = buf[-17]
    x_n17 = buf[-18]
    x_n32 = buf[-33]

    if self.is_fixed_point:
      part_a = -(int(x_n) >> 5) + int(x_n16) - int(x_n17) + (int(x_n32) >> 5)
    else:
      part_a = -(x_n/32) + x_n16 - x_n17 + (x_n32/32)

    y_n = hp_y1 + part_a

    if lead == 0:
      self.hp_y1_l0 = y_n
    else:
      self.hp_y1_l1 = y_n

    if rec:
      rec.record(y_n)
    return y_n

  def _run_dv_kernel(self, hp_sample, lead: int) -> int | float:
    buf = self.hp_buffer_l0 if lead == 0 else self.hp_buffer_l1
    rec = self.dv_recorder_l0 if lead == 0 else self.dv_recorder_l1

    buf.push(hp_sample)
    x_n  = buf[-1]
    x_n1 = buf[-2]
    x_n3 = buf[-4]
    x_n4 = buf[-5]

    if self.is_fixed_point:
      y_n = int(2*int(x_n) + int(x_n1) - int(x_n3) - 2*int(x_n4)) >> 3
    else:
      y_n = (1/8) * (2*x_n + x_n1 - x_n3 - 2*x_n4)

    if rec:
      rec.record(y_n)
    return y_n

  # --- Handoff for both leads ---

  def _handoff_both(self) -> None:
    # Try to hand off lead 0 output
    if self.output_data_l0 is not None:
      if self.next_unit is not None and self.next_unit.is_available():
        self.next_unit.input_data = self.output_data_l0
        self.output_data_l0 = None

    # Try to hand off lead 1 output
    if self.output_data_l1 is not None:
      if self.next_unit_l1 is not None and self.next_unit_l1.is_available():
        self.next_unit_l1.input_data = self.output_data_l1
        self.output_data_l1 = None

  # --- State machine tick ---

  def tick(self, current_cycle: int) -> None:
    # Try to hand off any pending outputs first
    self._handoff_both()

    # Check if either output is still pending — count as output stall
    if self.output_data_l0 is not None or self.output_data_l1 is not None:
      self.output_stall_cycles += 1
      self.stalled_cycles += 1
      return

    if self.state == IDLE:
      # Need both FIFOs to have data to start a new sample pair
      if self.fifo0.has_data() and self.fifo1.has_data():
        self._current_sample_l0 = self.fifo0.pop()
        self._current_sample_l1 = self.fifo1.pop()

        if self.raw_recorder_l0:
          self.raw_recorder_l0.record(self._current_sample_l0)
        if self.raw_recorder_l1:
          self.raw_recorder_l1.record(self._current_sample_l1)

        self.state = LP_BUSY_L0
        self.kernel_cycles_remaining = self.lp_cycles

      elif self.fifo0.has_data() or self.fifo1.has_data():
        # One FIFO has data but the other doesn't — structural hazard stall
        self.fifo_stall_cycles += 1
        self.stalled_cycles += 1

      else:
        # Both FIFOs empty — waiting for ADC
        self.idle_cycles += 1

    elif self.state == LP_BUSY_L0:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self._lp_result_l0 = self._run_lp_kernel(self._current_sample_l0, lead=0)
        self.state = HP_BUSY_L0
        self.kernel_cycles_remaining = self.hp_cycles

    elif self.state == HP_BUSY_L0:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self._hp_result_l0 = self._run_hp_kernel(self._lp_result_l0, lead=0)
        self.state = DV_BUSY_L0
        self.kernel_cycles_remaining = self.dv_cycles

    elif self.state == DV_BUSY_L0:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self.output_data_l0 = self._run_dv_kernel(self._hp_result_l0, lead=0)
        self.state = LP_BUSY_L1
        self.kernel_cycles_remaining = self.lp_cycles

    elif self.state == LP_BUSY_L1:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self._lp_result_l1 = self._run_lp_kernel(self._current_sample_l1, lead=1)
        self.state = HP_BUSY_L1
        self.kernel_cycles_remaining = self.hp_cycles

    elif self.state == HP_BUSY_L1:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self._hp_result_l1 = self._run_hp_kernel(self._lp_result_l1, lead=1)
        self.state = DV_BUSY_L1
        self.kernel_cycles_remaining = self.dv_cycles

    elif self.state == DV_BUSY_L1:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1
      if self.kernel_cycles_remaining == 0:
        self.output_data_l1 = self._run_dv_kernel(self._hp_result_l1, lead=1)
        self.state = IDLE

        self._handoff_both()

        if self.output_data_l0 is not None or self.output_data_l1 is not None:
          self.output_stall_cycles += 1
          self.stalled_cycles += 1

  def is_available(self) -> bool:
    return self.state == IDLE and self.output_data_l0 is None and self.output_data_l1 is None

  def is_stalled(self) -> bool:
    return self.output_data_l0 is not None or self.output_data_l1 is not None

  def compute(self, data):
    return data

  def __repr__(self) -> str:
    return (
      f"<SharedMACUnit name={self.name} "
      f"state={self.state} "
      f"busy_cycles={self.busy_cycles} "
      f"idle_cycles={self.idle_cycles} "
      f"stalled_cycles={self.stalled_cycles} "
      f"fifo_stall_cycles={self.fifo_stall_cycles} "
      f"output_stall_cycles={self.output_stall_cycles} "
      f"utilization={self.utilization:.1%}>"
    )
