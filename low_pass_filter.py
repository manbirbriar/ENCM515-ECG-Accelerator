import numpy as np
from hardware_unit import HardwareUnit

class LowPassFilter(HardwareUnit):
  INPUT_HISTORY = 12
  OUTPUT_HISTORY = 2

  def __init__(self, name: str, is_fixed: bool = False, latency_cycles: int = 4):
    super().__init__(name, latency_cycles=latency_cycles)

    self.is_fixed = is_fixed

    if is_fixed:
      self.input_history = np.zeros(self.INPUT_HISTORY, dtype=np.int32)
      self.output_history = np.zeros(self.OUTPUT_HISTORY, dtype=np.int32)
    else:
      self.input_history = np.zeros(self.INPUT_HISTORY, dtype=np.float32)
      self.output_history = np.zeros(self.OUTPUT_HISTORY, dtype=np.float32)

  def compute(self, data: list) -> list:
    x = np.array(data, dtype=np.int32 if self.is_fixed else np.float32)
    n = len(x)
    y = np.zeros(n, dtype=np.int32 if self.is_fixed else np.float32)

    x_padded = np.concatenate([self.input_history, x])

    for i in range(n):
      xi = x_padded[i + self.INPUT_HISTORY] # x[n]
      xi_6 = x_padded[i + self.INPUT_HISTORY - 6] # x[n-6]
      xi_12 = x_padded[i + self.INPUT_HISTORY - 12] # x[n-12]

      yi_1 = y[i - 1] if i >= 1 else self.output_history[1] # y[n-1]
      yi_2 = y[i - 2] if i >= 2 else (y[i - 1] if i == 1 else self.output_history[0]) # y[n-2]

      y[i] = 2 * yi_1 - yi_2 + xi - 2 * xi_6 + xi_12

    self.input_history = x_padded[-self.INPUT_HISTORY:]
    self.output_history[0] = y[-2]
    self.output_history[1] = y[-1]

    return y.tolist()

  def __repr__(self) -> str:
    return f"<LowPassFilter name={self.name} is_fixed={self.is_fixed} latency_cycles={self.latency_cycles} busy={self.busy}>"