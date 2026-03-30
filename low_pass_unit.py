import numpy as np
from hardware_unit import HardwareUnit
class LowPassUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int, is_fixed_point: bool):
    # 1. FIR Stage: (x[n] - 2x[n-6] + x[n-12])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 2 ops (sub, add) per sample. 2* is a "free" bit-shift.
    fir_cycles = (2 * window_size) // vector_width

    # 2. IIR Stage: (2y[n-1] - y[n-2] + part_a)
    #    - Sequential Bottleneck: y[n] has a dependency on y[n-1] and y[n-2].
    #    - Cost: 2 ops (sub, add) per sample. 2* is a "free" bit-shift.
    # iir_cycles = 2 * window_size
    iir_cycles = window_size

    latency = fir_cycles + iir_cycles

    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)
    
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