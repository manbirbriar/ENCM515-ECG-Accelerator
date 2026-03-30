import numpy as np
from hardware_unit import HardwareUnit

class DerivativeUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 3 ops (add, sub, sub) per sample. (1/8)* is a "free" bit-shift.
    fir_cycles = (3 * window_size) // vector_width
    latency = fir_cycles

    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)
    
    self.window_size = window_size
    self.vector_width = vector_width

    # internal registers
    self.x_history = np.zeros(4) # x[n-4]

  # y[n] = (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    if self.is_fixed_point:
      results = (2 * x_extended[4:] + x_extended[3:-1] - x_extended[1:-3] - 2 * x_extended[:self.window_size])
      # y[n] = [2x[n] + x[n-1] - x[n-3] - 2x[n-4]] >> 3
      results = results.astype(np.int32) >> 3
    else:
      # y[n] = (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
      results = (1/8) * (2 * x_extended[4:] + x_extended[3:-1] - x_extended[1:-3] - 2 * x_extended[:self.window_size])

    self.x_history = x_current[-4:]
    
    return results.tolist()