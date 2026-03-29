import numpy as np
from hardware_unit import HardwareUnit

class MWIUnit(HardwareUnit):
  def __init__(self, name: str, window_size: int, vector_width: int, mwi_window_size: int):
    # Latency Model (shifts treated as free):
    latency = int(np.ceil(np.log2(mwi_window_size)))

    super().__init__(name, latency_cycles=latency)
    
    self.window_size = window_size
    self.vector_width = vector_width
    self.mwi_window_size = mwi_window_size
    
    # internal registers
    self.x_history = np.zeros(mwi_window_size)

  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    results = []

    for i in range(len(x_current)):
      window_sum = np.sum(x_extended[i : i + self.mwi_window_size])
      results.append(window_sum * (1 / self.mwi_window_size))

    self.x_history = x_current[-self.mwi_window_size:]

    return results

  def __repr__(self) -> str:
    return f"<MWIUnit name={self.name} latency={self.latency_cycles} busy={self.busy}>"