import numpy as np
from hardware_unit import HardwareUnit

class DerivativeUnit(HardwareUnit):

  def __init__(self, name: str, vector_width: int = 4):
    # TODO: Confirm this number is correct
    # 72 // 4 = 18 cycles
    latency = (72 // vector_width)

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

    # internal registers
    self.x_history = np.zeros(4) # previous inputs x[n-4]

  # y[n] = (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    # 4 ALUs calculating slopes simultaneously
    results = (1/8) * (2 * x_extended[4:] + x_extended[3:-1] - x_extended[1:-3] - 2 * x_extended[:72])

    self.x_history = x_current[-4:]
    
    return results.tolist()

  def __repr__(self) -> str:
    return f"<DerivativeUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"