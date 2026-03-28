import numpy as np
from hardware_unit import HardwareUnit

class LowPassUnit(HardwareUnit):

  def __init__(self, name: str, vector_width: int = 4):
    # TODO: Confirm this number is correct
    # 18 FIR SIMD cycles + 72 IIR cycles = 90 cycles
    latency = (72 // vector_width) + 72

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

    # internal registers
    self.y_1 = 0.0 # most recent output y[n-1]
    self.y_2 = 0.0 # second most recent output y[n-2]
    self.x_history = np.zeros(12) # previous inputs x[n-12]

  # y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    # part_a = x[n] - 2x[n-6] + x[n-12]
    part_a = (x_extended[12:] - 2 * x_extended[6:-6] + x_extended[:-12])

    results = []
    for i in range(len(part_a)):
        
      y_n = (2 * self.y_1) - self.y_2 + part_a[i]
      
      self.y_2 = self.y_1
      self.y_1 = y_n
      
      results.append(y_n)

    self.x_history = x_current[-12:]
    
    return results

  def __repr__(self) -> str:
    return f"<LowPassUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"