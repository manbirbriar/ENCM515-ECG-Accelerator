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

class LowPassUnit(HardwareUnit):
  
  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: (x[n] - 2x[n-6] + x[n-12])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Float ops/sample: mul + sub + add
    #    - Fixed ops/sample: shift + sub + add
    float_fir_ops_per_sample = FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES
    fixed_fir_ops_per_sample = FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + FIXED_ADD_CYCLES
    
    fir_float_cycles = (window_size * float_fir_ops_per_sample + vector_width - 1) // vector_width
    fir_fixed_cycles = (window_size * fixed_fir_ops_per_sample + vector_width - 1) // vector_width

    # 2. IIR Stage: (2y[n-1] - y[n-2] + part_a)
    #    - Sequential Bottleneck: y[n] has a dependency on y[n-1] and y[n-2].
    #    - Float ops/sample: mul + sub + add
    #    - Fixed ops/sample: shift + sub + add
    iir_float_cycles = window_size * (FLOAT_MUL_CYCLES + FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES)
    iir_fixed_cycles = window_size * (FIXED_SHIFT_CYCLES + FIXED_SUB_CYCLES + FIXED_ADD_CYCLES)

    fixed_latency = fir_fixed_cycles + iir_fixed_cycles
    float_latency = fir_float_cycles + iir_float_cycles

    if is_fixed_point:
      super().__init__(name, latency_cycles=fixed_latency, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=float_latency, is_fixed_point=is_fixed_point)
    
    self.vector_width = vector_width

    # internal registers
    self.y_1 = 0.0 # y[n-1]
    self.y_2 = 0.0 # y[n-2]
    self.x_history = np.zeros(12) # x[n-12]

  # y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    if self.is_fixed_point:
      # part_a = x[n] - x[n-6] << 1 + x[n-12]
      part_a = x_extended[12:] - (x_extended[6:-6].astype(np.int32) << 1) + x_extended[:-12]
    else:
      # part_a = x[n] - 2x[n-6] + x[n-12]
      part_a = (x_extended[12:] - 2 * x_extended[6:-6] + x_extended[:-12])

    results = []
    for i in range(len(part_a)):
      if self.is_fixed_point:
        # y[n] = y[n-1] << 1 - y[n-2] + part_a
        y_n = (int(self.y_1) << 1) - int(self.y_2) + int(part_a[i])
      else:
        # y[n] = 2y[n-1] - y[n-2] + part_a
        y_n = (2 * self.y_1) - self.y_2 + part_a[i]
      
      self.y_2, self.y_1 = self.y_1, y_n
      results.append(y_n)

    self.x_history = x_current[-12:]
    return results