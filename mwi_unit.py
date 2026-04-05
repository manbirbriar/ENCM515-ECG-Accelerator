import numpy as np
from config import FIXED_ADD_CYCLES, FIXED_MUL_CYCLES, FIXED_SHIFT_CYCLES, FIXED_SUB_CYCLES, FLOAT_ADD_CYCLES, FLOAT_MUL_CYCLES, FLOAT_SUB_CYCLES, MWI_WINDOW_SIZE
from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer

# Magic number approximation for fixed-point division by MWI_WINDOW_SIZE (54)
MWI_NORM_MAGIC: int = 4855
MWI_NORM_SHIFT: int = 18

class MWIUnit(HardwareUnit):
  """
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4122029

  Moving Window Integrator: y[n] = (1/N) * sum(x[n-N+1] ... x[n])

  Hardware: 1 ALU, 1 Circular Buffer, and 1 Registers (Running Sum)
  """
  def __init__(self, name: str, is_fixed_point: bool):
    # 1 SUB, 1 ADD, 1 MUL, and 1 SHIFT operation
    if is_fixed_point:
      latency = FIXED_SUB_CYCLES + FIXED_ADD_CYCLES + FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES
    else:
      # 1 SUB, 1 ADD, and 1 MUL operation
      latency = FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES + FLOAT_MUL_CYCLES

    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.mwi_window_size = MWI_WINDOW_SIZE
    self.norm_constant = 1.0 / MWI_WINDOW_SIZE

    self.window_buffer = CircularBuffer(MWI_WINDOW_SIZE, dtype=int if is_fixed_point else float)
    self.running_sum = 0

  def compute(self, sample) -> int | float:
    oldest = self.window_buffer[-self.mwi_window_size]
    self.running_sum += sample - oldest
    self.window_buffer.push(sample)

    if self.is_fixed_point:
      # int64 to prevent overflow during intermediate multiplication
      return int((np.int64(self.running_sum) * MWI_NORM_MAGIC) >> MWI_NORM_SHIFT)
    else:
      return self.running_sum * self.norm_constant