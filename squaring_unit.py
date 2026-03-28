import numpy as np
from hardware_unit import HardwareUnit

class SquaringUnit(HardwareUnit):

  def __init__(self, name: str, vector_width: int = 4):
    # 72 // 4 = 18 cycles
    latency = (72 // vector_width)

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

  # y[n] = x[n]^2
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    
    # 4 ALUs squaring 4 samples at once
    results = np.square(x_current)

    return results.tolist()

  def __repr__(self) -> str:
    return f"<SquaringUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"