import numpy as np
from hardware_unit import HardwareUnit
from config import FIXED_POINT_BITS, FIXED_MUL_CYCLES, FIXED_SHIFT_CYCLES, FLOAT_MUL_CYCLES

class SquaringUnit(HardwareUnit):
  """
  Dedicated Squarer: y[n] = x[n]^2
  No history or state.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    if is_fixed_point:
      # Requires a right shift to rescale into Q15 format after squaring.
      latency = FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES
    else:
      latency = FLOAT_MUL_CYCLES
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

  def compute(self, sample) -> int | float:
    if self.is_fixed_point:
      return int((np.int64(sample) * np.int64(sample)) >> FIXED_POINT_BITS)
    else:
      return sample * sample