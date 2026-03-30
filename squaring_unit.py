import numpy as np
from hardware_unit import HardwareUnit
from config import FIXED_POINT_BITS, FIXED_MUL_CYCLES, FIXED_SHIFT_CYCLES, FLOAT_MUL_CYCLES

class SquaringUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: x[n]^2
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Float ops/sample: mul
    #    - Fixed ops/sample: mul + shift (Q-format rescale)
    float_fir_ops_per_sample = FLOAT_MUL_CYCLES
    fixed_fir_ops_per_sample = FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES
    fir_float_cycles = (window_size * float_fir_ops_per_sample + vector_width - 1) // vector_width
    fir_fixed_cycles = (window_size * fixed_fir_ops_per_sample + vector_width - 1) // vector_width

    if is_fixed_point:
      super().__init__(name, latency_cycles=fir_fixed_cycles, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=fir_float_cycles, is_fixed_point=is_fixed_point)
    
    self.vector_width = vector_width

  # y[n] = x[n]^2
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    
    if self.is_fixed_point:
      # int64 to prevent overflow
      # y[n] = x[n]^2 >> 15
      results = ((x_current.astype(np.int64)**2) >> FIXED_POINT_BITS).astype(np.int32)
    else:
      # y[n] = x[n]^2
      results = np.square(x_current)

    return results.tolist()