# from hardware_unit import HardwareUnit
# import numpy as np

# class DerivativeUnit(HardwareUnit):
#   def __init__(self, name: str, latency_cycles: int = 1, scale: float = 1.0):
#     super().__init__(name, latency_cycles)
#     self.scale = scale

#     # self.history: list[list[float]] = [] # storage for saving history for visualization

#   def compute(self, data: list | np.ndarray) -> list:
#     x = np.asarray(data, dtype=np.float64)
#     y = np.zeros_like(x)

#     # causal 5-point derivative approximation
#     for n in range(4, len(x)):
#       y[n] = self.scale * (
#         2 * x[n]
#         + x[n - 1]
#         - x[n - 3]
#         - 2 * x[n - 4]
#       )

#     result = y.tolist()
#     # self.history.append(result)
#     return result

import numpy as np
from hardware_unit import HardwareUnit

# TODO: This is temporary (AI)
class DerivativeUnit(HardwareUnit):

  def __init__(self, name: str, vector_width: int = 4):
    # TODO: Confirm this number is correct
    # Pure FIR logic: 72 // 4 = 18 cycles. No sequential IIR bottleneck!
    latency = (72 // vector_width)

    super().__init__(name, latency_cycles=latency)
    
    self.vector_width = vector_width

    # internal registers
    self.x_history = np.zeros(4) # previous inputs x[n-4]

  # y[n] = (1/8) * [2x[n] + x[n-1] - x[n-3] - 2x[n-4]]
  def compute(self, data: list) -> list:
    x_current = np.array(data)
    x_extended = np.concatenate([self.x_history, x_current])
    
    # --- PHASE 1: FULLY VECTORIZED ---
    # This represents 4 ALUs calculating slopes simultaneously
    # x_extended[4:] is x[n]
    # x_extended[3:-1] is x[n-1]
    # x_extended[1:-3] is x[n-3]
    # x_extended[:72] is x[n-4]
    results = (1/8) * (2 * x_extended[4:] + x_extended[3:-1] - x_extended[1:-3] - 2 * x_extended[:72])

    self.x_history = x_current[-4:]
    
    return results.tolist()

  def __repr__(self) -> str:
    return f"<DerivativeUnit name={self.name} latency_cycles={self.latency_cycles} busy={self.busy}>"