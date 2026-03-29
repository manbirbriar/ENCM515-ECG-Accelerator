import numpy as np
from hardware_unit import HardwareUnit

class HighPassUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int):

    # 1. FIR Stage: (-(1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 3 ops (sub, add, add) per sample. (1/32)* is a "free" bit-shift.
    fir_cycles = (3 * window_size) // vector_width

    # 2. IIR Stage: (y[n-1] + part_a)
    #    - Sequential Bottleneck: y[n] has a dependency on y[n-1].
    #    - Cost: 1 op (add) per sample.
    iir_cycles = 1 * window_size
    latency = fir_cycles + iir_cycles

    super().__init__(name, latency_cycles=latency)
    
    self.window_size = window_size
    self.vector_width = vector_width

    # internal registers
    self.y_1 = 0.0 # y[n-1]
    self.x_history = np.zeros(32) # x[n-32]

  # y[n] = y[n-1] - (1/32)x[n] + x[n-16] - x[n-17] + (1/32)x[n-32]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
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

  def __repr__(self) -> str:
    return f"<HighPassUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"