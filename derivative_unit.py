import numpy as np
from hardware_unit import HardwareUnit
from config import (
  FIXED_ADD_CYCLES,
  FIXED_SUB_CYCLES,
  FIXED_SHIFT_CYCLES,
  FLOAT_ADD_CYCLES,
  FLOAT_SUB_CYCLES,
  FLOAT_MUL_CYCLES,
)

class DerivativeUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Float ops/sample: 2 mul + 1 add + 2 sub
    #    - Fixed ops/sample: 3 shift + 1 add + 2 sub
    float_fir_ops_per_sample = (2 * FLOAT_MUL_CYCLES) + FLOAT_ADD_CYCLES + (2 * FLOAT_SUB_CYCLES)
    fixed_fir_ops_per_sample = (3 * FIXED_SHIFT_CYCLES) + FIXED_ADD_CYCLES + (2 * FIXED_SUB_CYCLES)
    fir_float_cycles = (window_size * float_fir_ops_per_sample + vector_width - 1) // vector_width
    fir_fixed_cycles = (window_size * fixed_fir_ops_per_sample + vector_width - 1) // vector_width

    if is_fixed_point:
      super().__init__(name, latency_cycles=fir_fixed_cycles, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=fir_float_cycles, is_fixed_point=is_fixed_point)
    
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