# from hardware_unit import HardwareUnit
# import numpy as np

# class SquaringUnit(HardwareUnit):
#   def __init__(self, name: str, latency_cycles: int = 1):
#     super().__init__(name, latency_cycles)

#     # self.history: list[list[float]] = [] # storage for saving history for visualization

#   def compute(self, data: list | np.ndarray) -> list:
#     x = np.asarray(data, dtype=np.float64)
#     y = np.square(x)

#     result = y.tolist()
#     # self.history.append(result)
#     return result

import numpy as np
from hardware_unit import HardwareUnit

# TODO: This is temporary (AI)
class SquaringUnit(HardwareUnit):

  def __init__(self, name: str, vector_width: int = 4):
    # This is a pure FIR/pointwise operation: (72 / 4) = 18 cycles
    # No IIR feedback bottleneck here!
    latency = (72 // vector_width)

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

  # y[n] = x[n]^2
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    
    # --- PHASE 1: FULLY VECTORIZED ---
    # In hardware, this represents 4 ALUs squaring 4 samples at once
    results = np.square(x_current)

    # Convert back to list for the next pipeline stage
    return results.tolist()

  def __repr__(self) -> str:
    return f"<SquaringUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"