import numpy as np
from hardware_unit import HardwareUnit
from config import FIXED_POINT_BITS

class SquaringUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: x[n]^2
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 1 op (mul) per sample.
    fir_cycles = (1 * window_size) // vector_width
    latency = fir_cycles

    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)
    
    self.vector_width = vector_width

  # y[n] = x[n]^2
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    
    if self.is_fixed_point:
      # int64 to prevent overflow
      # y[n] = x[n]^2 >> 15
      results = (x_current.astype(np.int64)**2) >> FIXED_POINT_BITS
    else:
      # y[n] = x[n]^2
      results = np.square(x_current)

    return results.tolist()