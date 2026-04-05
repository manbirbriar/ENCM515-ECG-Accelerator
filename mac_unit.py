from __future__ import annotations
from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer
from fifo_buffer import FIFOBuffer
from data_recorder import DataRecorder
from config import FIXED_MAC_CYCLES, FLOAT_MAC_CYCLES

# State Machine States
IDLE = "IDLE"
LP_BUSY = "LP_BUSY"
HP_BUSY = "HP_BUSY"
DV_BUSY = "DV_BUSY"

class MACUnit(HardwareUnit):
  """
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4122029

  MAC unit that runs three FIR/IIR filter kernels sequentially using MAC operations.
  A time-multiplexed design was chosen over three dedicated MACs to minimise hardware area.
  Coefficients are stored in a Coefficient ROM inside this hardware unit.

  1.
  Low-pass filter: y[n] = 2*y[n-1] - y[n-2] + x[n] - 2*x[n-6] + x[n-12]
  Coefficients: [2, -1, 1, -2, 1]

  2.
  High-pass filter: y[n] = 32*x[n-16] - y[n-1] - x[n] + x[n-32]
  Coefficients: [32, -1, -1, 1]

  3.
  Derivative filter: y[n] = (1/8)*[2*x[n] + x[n-1] - x[n-3] - 2*x[n-4]]
  Coefficients: [2/8, 1/8, -1/8, -2/8]

  Circular history buffers:
  1. lp_buffer[13]
  2. hp_buffer[33]
  3. dv_buffer[5]

  IIR state registers:
  LowPass: lp_y1, lp_y2
  HighPass: hp_y1
  """
  def __init__(self, name: str, fifo: FIFOBuffer, is_fixed_point: bool):
    if is_fixed_point:
      # LowPass: 5 MAC operations (2 IIR + 3 FIR)
      self.lp_cycles = 5 * FIXED_MAC_CYCLES
      # HighPass: 4 MAC operations (1 IIR + 3 FIR)
      self.hp_cycles = 4 * FIXED_MAC_CYCLES
      # Derivative: 4 MAC operations (4 FIR)
      self.dv_cycles = 4 * FIXED_MAC_CYCLES
    else:
      # LowPass: 5 MAC operations (2 IIR + 3 FIR)
      self.lp_cycles = 5 * FLOAT_MAC_CYCLES
      # HighPass: 4 MAC operations (1 IIR + 3 FIR)
      self.hp_cycles = 4 * FLOAT_MAC_CYCLES
      # Derivative: 4 MAC operations (4 FIR)
      self.dv_cycles = 4 * FLOAT_MAC_CYCLES

    total_latency = self.lp_cycles + self.hp_cycles + self.dv_cycles
    super().__init__(name, latency_cycles=total_latency, is_fixed_point=is_fixed_point)

    self.fifo = fifo

    # History buffers (Shift Registers)
    self.lp_buffer = CircularBuffer(13, dtype=int if is_fixed_point else float)
    self.hp_buffer = CircularBuffer(33, dtype=int if is_fixed_point else float)
    self.dv_buffer = CircularBuffer(5,  dtype=int if is_fixed_point else float)

    # IIR State Registers
    self.lp_y1 = 0
    self.lp_y2 = 0
    self.hp_y1 = 0

    # Internal State Machine
    self.state = IDLE
    self.kernel_cycles_remaining = 0

    # Intermediate Kernel Registers
    self.current_sample = None
    self.lp_result = None
    self.hp_result = None

    self.raw_recorder: DataRecorder | None = None
    self.lp_recorder: DataRecorder | None = None
    self.hp_recorder: DataRecorder | None = None
    self.dv_recorder: DataRecorder | None = None

  def attach_raw_recorder(self, recorder: DataRecorder) -> None:
    self.raw_recorder = recorder
  def attach_lp_recorder(self, recorder: DataRecorder) -> None:
    self.lp_recorder = recorder
  def attach_hp_recorder(self, recorder: DataRecorder) -> None:
    self.hp_recorder = recorder
  def attach_dv_recorder(self, recorder: DataRecorder) -> None:
    self.dv_recorder = recorder

  def run_lp_kernel(self, sample) -> int | float:
    """
    Low-Pass Filter using MAC operations:
    y[n] = 2*y[n-1] - y[n-2] + x[n] - 2*x[n-6] + x[n-12]
    
    accumulator = 0
    accumulator += 2 * y[n-1]
    accumulator += -1 * y[n-2]
    accumulator += 1 * x[n]
    accumulator += -2 * x[n-6]
    accumulator += 1 * x[n-12]

    y[n] = accumulator
    """
    self.lp_buffer.push(sample)

    x_n = self.lp_buffer[-1]
    x_n6 = self.lp_buffer[-7]
    x_n12 = self.lp_buffer[-13]

    accumulator = 0
    if self.is_fixed_point:
      accumulator += 2 * int(self.lp_y1)
      accumulator += -1 * int(self.lp_y2)
      accumulator += 1 * int(x_n)
      accumulator += -2 * int(x_n6)
      accumulator += 1 * int(x_n12)

      y_n = int(accumulator)
    else:
      accumulator += 2.0 * self.lp_y1
      accumulator += -1.0 * self.lp_y2
      accumulator += 1.0 * x_n
      accumulator += -2.0 * x_n6
      accumulator += 1.0 * x_n12

      y_n = float(accumulator)

    self.lp_y2 = self.lp_y1
    self.lp_y1 = y_n

    self.lp_recorder.record(y_n)

    return y_n

  def run_hp_kernel(self, lp_sample) -> int | float:
    """
    High-Pass Filter using MAC operations:
    y[n] = 32*x[n-16] - y[n-1] - x[n] + x[n-32]
    
    accumulator = 0
    accumulator += 32 * x[n-16]
    accumulator += -1 * y[n-1]
    accumulator += -1 * x[n]
    accumulator += 1 * x[n-32]

    y[n] = accumulator
    """
    self.hp_buffer.push(lp_sample)

    x_n = self.hp_buffer[-1]
    x_n16 = self.hp_buffer[-17]
    x_n32 = self.hp_buffer[-33]

    accumulator = 0
    if self.is_fixed_point:
      accumulator += 1 * (int(x_n16) << 5)
      accumulator += -1 * int(self.hp_y1)
      accumulator += -1 * int(x_n)
      accumulator += 1 * int(x_n32)

      y_n = int(accumulator)
    else:
      accumulator += 32.0 * x_n16
      accumulator += -1.0 * self.hp_y1
      accumulator += -1.0 * x_n
      accumulator += 1.0 * x_n32
      
      y_n = float(accumulator)

    self.hp_y1 = y_n

    self.hp_recorder.record(y_n)

    return y_n

  def run_dv_kernel(self, hp_sample) -> int | float:
    """
    Derivative Filter using MAC operations:
    y[n] = (1/8)*[2*x[n] + x[n-1] - x[n-3] - 2*x[n-4]]
    
    accumulator = 0
    accumulator += 2/8 * x[n]
    accumulator += 1/8 * x[n-1]
    accumulator += -1/8 * x[n-3]
    accumulator += -2/8 * x[n-4]
    
    y[n] = accumulator
    """
    self.dv_buffer.push(hp_sample)

    x_n = self.dv_buffer[-1]
    x_n1 = self.dv_buffer[-2]
    x_n3 = self.dv_buffer[-4]
    x_n4 = self.dv_buffer[-5]

    accumulator = 0
    if self.is_fixed_point:
      accumulator += (2 * int(x_n)) >> 3
      accumulator += int(x_n1) >> 3
      accumulator += -(int(x_n3) >> 3)
      accumulator += -(2 * int(x_n4)) >> 3

      y_n = int(accumulator)
    else:
      accumulator += (2.0 * x_n) / 8.0
      accumulator += x_n1 / 8.0
      accumulator += -(x_n3 / 8.0)
      accumulator += -(2.0 * x_n4) / 8.0

      y_n = float(accumulator)

    self.dv_recorder.record(y_n)

    return y_n

  # Overrides HardwareUnit.tick() to implement the internal state machine
  def tick(self, current_cycle: int) -> None:
    # Try to handoff any pending output first
    # Handles the case where successor became available this cycle
    if self.output_data is not None:
      self.handoff_to_next()

    # If output is still pending after handoff attempt, we are stalled
    if self.output_data is not None:
      self.stalled_cycles += 1
      return

    if self.state == IDLE:
      # Take sample from FIFO
      if self.fifo.has_data():
        sample = self.fifo.pop()

        self.raw_recorder.record(sample)

        self.current_sample = sample
        self.state = LP_BUSY
        self.kernel_cycles_remaining = self.lp_cycles
      else:
        self.idle_cycles += 1

    elif self.state == LP_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        # Pass to Low Pass
        self.lp_result = self.run_lp_kernel(self.current_sample)
        self.state = HP_BUSY
        self.kernel_cycles_remaining = self.hp_cycles

    elif self.state == HP_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        # Pass to High Pass
        self.hp_result = self.run_hp_kernel(self.lp_result)
        self.state = DV_BUSY
        self.kernel_cycles_remaining = self.dv_cycles

    elif self.state == DV_BUSY:
      self.busy_cycles += 1
      self.kernel_cycles_remaining -= 1

      if self.kernel_cycles_remaining == 0:
        # Pass to Derivative
        dv_result = self.run_dv_kernel(self.hp_result)
        self.output_data = dv_result
        self.state = IDLE

        self.handoff_to_next()

        # If handoff failed, successor was busy so we stall
        if self.output_data is not None:
          self.stalled_cycles += 1

  def is_available(self) -> bool:
    return self.state == IDLE and self.output_data is None

  def is_stalled(self) -> bool:
    return self.output_data is not None

  def compute(self, data):
    return data