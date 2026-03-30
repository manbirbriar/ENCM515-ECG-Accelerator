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

class HighPassUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):

    # 1. FIR Stage: (-(1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Float ops/sample: 2 mul + 1 sub + 2 add
    #    - Fixed ops/sample: 2 shift + 1 sub + 2 add
    float_fir_ops_per_sample = (2 * FLOAT_MUL_CYCLES) + FLOAT_SUB_CYCLES + (2 * FLOAT_ADD_CYCLES)
    fixed_fir_ops_per_sample = (2 * FIXED_SHIFT_CYCLES) + FIXED_SUB_CYCLES + (2 * FIXED_ADD_CYCLES)
    fir_float_cycles = (window_size * float_fir_ops_per_sample + vector_width - 1) // vector_width
    fir_fixed_cycles = (window_size * fixed_fir_ops_per_sample + vector_width - 1) // vector_width

    # 2. IIR Stage: (y[n-1] + part_a)
    #    - Sequential Bottleneck: y[n] has a dependency on y[n-1].
    #    - Cost: 1 add per sample.
    iir_float_cycles = window_size * FLOAT_ADD_CYCLES
    iir_fixed_cycles = window_size * FIXED_ADD_CYCLES

    float_latency = fir_float_cycles + iir_float_cycles
    fixed_latency = fir_fixed_cycles + iir_fixed_cycles

    if is_fixed_point:
      super().__init__(name, latency_cycles=fixed_latency, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=float_latency, is_fixed_point=is_fixed_point)
    
    self.window_size = window_size
    self.vector_width = vector_width

    # internal registers
    self.y_1 = 0.0 # y[n-1]
    self.x_history = np.zeros(32) # x[n-32]

  # y[n] = y[n-1] - (1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    if self.is_fixed_point:
      x_n = x_extended[32:].astype(np.int32)
      x_32 = x_extended[:self.window_size].astype(np.int32)
      # part_a = -(1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
      part_a = -(x_n >> 5) + x_extended[16:-16] - x_extended[15:-17] + (x_32 >> 5)
    else:
      # part_a = -(1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
      part_a = (-(1/32) * x_extended[32:] + x_extended[16:-16] - x_extended[15:-17] + (1/32) * x_extended[:self.window_size])

    results = []
    for i in range(len(part_a)):
      # y[n] = y[n-1] + part_a
      y_n = self.y_1 + part_a[i]
      self.y_1 = y_n
      results.append(y_n)

    self.x_history = x_current[-32:]
    return results