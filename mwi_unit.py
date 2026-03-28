import numpy as np
from hardware_unit import HardwareUnit

# TODO: This is temporary (AI)
class MWIUnit(HardwareUnit):
  def __init__(self, name: str, vector_width: int = 4, mwi_window: int = 54):
    # Performance: log2(window_size) cycles for a parallel reduction sum
    latency = int(np.ceil(np.log2(mwi_window)))
    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width
    self.mwi_window = mwi_window
    
    # State register: We need the last 53 samples to calculate the first 
    # sum of the next 72-sample window.
    self.x_history = np.zeros(mwi_window - 1)

  def compute(self, data: list) -> list:
    x_current = np.array(data)
    # Stitched signal for seamless windowing
    x_ext = np.concatenate([self.x_history, x_current])
    
    results = []
    # In hardware, this loop is a single-cycle 'sliding window' or 
    # a highly parallel adder tree.
    for i in range(len(x_current)):
      window_sum = np.sum(x_ext[i : i + self.mwi_window])
      # We divide by window size to normalize the pulse height
      results.append(window_sum / self.mwi_window)

    # Save the 'tail' for the next cycle's history
    self.x_history = x_current[-(self.mwi_window - 1):]
    return results

  def __repr__(self) -> str:
    return f"<MWIUnit name={self.name} latency={self.latency_cycles} busy={self.busy}>"