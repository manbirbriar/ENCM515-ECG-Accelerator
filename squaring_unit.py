import numpy as np
from hardware_unit import HardwareUnit

class SquaringUnit(HardwareUnit):

  def __init__(self, name: str, window_size: int, vector_width: int):
    # 1. FIR Stage: x[n]^2
    #    - Exploit DLP: We emulate SIMD vectorization where "vector_width" samples are processed in parallel.
    #    - Cost: 1 op (mul) per sample.
    fir_cycles = (1 * window_size) // vector_width
    latency = fir_cycles

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

  # y[n] = x[n]^2
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    
    results = np.square(x_current)

    return results.tolist()

  def __repr__(self) -> str:
    return f"<SquaringUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"