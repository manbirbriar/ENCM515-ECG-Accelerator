from hardware_unit import HardwareUnit
import numpy as np

class SquaringUnit(HardwareUnit):
    def __init__(self, name: str, latency_cycles: int = 1):
        super().__init__(name, latency_cycles)

        self.history: list[list[float]] = [] # storage for saving history for visualization

    def compute(self, data: list | np.ndarray) -> list:
        x = np.asarray(data, dtype=np.float64)
        y = np.square(x)

        result = y.tolist()
        self.history.append(result)
        return result