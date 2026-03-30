import numpy as np
from hardware_unit import HardwareUnit
from config import FIXED_ADD_CYCLES, FIXED_MUL_CYCLES, FIXED_SUB_CYCLES, FLOAT_ADD_CYCLES, FLOAT_MUL_CYCLES, FLOAT_SUB_CYCLES

class MWIUnit(HardwareUnit):
  def __init__(self, name: str, window_size: int, vector_width: int, mwi_window_size: int, is_fixed_point: bool):
    # 1. FIR Stage: (x[n] - x[n-mwi_window_size])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 1 op (sub) per sample.
    fir_float_cycles = (window_size * FLOAT_SUB_CYCLES + vector_width - 1) // vector_width
    fir_fixed_cycles = (window_size * FIXED_SUB_CYCLES + vector_width - 1) // vector_width

    # 2. IIR Stage: (sum[n] = sum[n-1] + part_a)
    #    - Sequential Bottleneck: sum[n] has a dependency on sum[n-1].
    #    - Cost: 1 op (add) per sample.
    iir_float_cycles = window_size * FLOAT_ADD_CYCLES
    iir_fixed_cycles = window_size * FIXED_ADD_CYCLES
    
    # 3. Normalization: (sum[n] * (1/mwi_window_size))
    #    - Cost: 1 op (mul) per sample.
    norm_float_cycles = window_size * FLOAT_MUL_CYCLES
    norm_fixed_cycles = window_size * FIXED_MUL_CYCLES

    float_latency = fir_float_cycles + iir_float_cycles + norm_float_cycles
    fixed_latency = fir_fixed_cycles + iir_fixed_cycles + norm_fixed_cycles

    if is_fixed_point:
      super().__init__(name, latency_cycles=fixed_latency, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=float_latency, is_fixed_point=is_fixed_point)
    
    self.window_size = window_size
    self.vector_width = vector_width
    self.mwi_window_size = mwi_window_size
    
    # internal registers
    self.running_sum = 0.0
    self.x_history = np.zeros(mwi_window_size)
    self.norm_constant = 1.0 / mwi_window_size

  # y[n] = (1/mwi_window_size) * [sum[n-1] + x[n] - x[n-mwi_window_size]]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])

    # part_a = x[n] - x[n-mwi_window_size]
    part_a = x_extended[self.mwi_window_size:] - x_extended[:self.window_size]
    
    results = []
    for i in range(len(x_current)):
      # sum[n] = sum[n-1] + part_a
      self.running_sum += part_a[i]
      
      if self.is_fixed_point:
        # y[n] = sum[n] // mwi_window_size
        y_n = self.running_sum // self.mwi_window_size
      else:
        # y[n] = (1/mwi_window_size) * sum[n]
        y_n = self.running_sum * self.norm_constant
      results.append(y_n)

    self.x_history = x_current[-self.mwi_window_size:]

    return results