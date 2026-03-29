import numpy as np
from hardware_unit import HardwareUnit

class MWIUnit(HardwareUnit):
  def __init__(self, name: str, window_size: int, vector_width: int, mwi_window_size: int):
    # 1. FIR Stage: (x[n] - x[n-mwi_window_size])
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 1 op (sub) per sample.
    fir_cycles = (1 * window_size) // vector_width

    # 2. IIR Stage: (sum[n] = sum[n-1] + part_a)
    #    - Sequential Bottleneck: sum[n] has a dependency on sum[n-1].
    #    - Cost: 1 op (add) per sample.
    iir_cycles = 1 * window_size
    
    # 3. Normalization: (sum[n] * (1/mwi_window_size))
    #    - Cost: 1 op (mul) per sample.
    norm_cycles = 1 * window_size

    latency = fir_cycles + iir_cycles + norm_cycles

    super().__init__(name, latency_cycles=latency)
    
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
      self.running_sum = self.running_sum + part_a[i]

      # y[n] = (1/mwi_window_size) * sum[n]
      y_n = self.running_sum * self.norm_constant

      results.append(y_n)

    self.x_history = x_current[-self.mwi_window_size:]

    return results

  def __repr__(self) -> str:
    return f"<MWIUnit name={self.name} latency={self.latency_cycles} busy={self.busy}>"