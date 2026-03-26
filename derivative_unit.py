from hardware_unit import HardwareUnit
import numpy as np

class DerivativeUnit(HardwareUnit):
    def __init__(self, name: str, latency_cycles: int = 1, scale: float = 1.0):
        super().__init__(name, latency_cycles)
        self.scale = scale

        self.history: list[list[float]] = [] # storage for saving history for visualization

    def compute(self, data: list | np.ndarray) -> list:
        x = np.asarray(data, dtype=np.float64)
        y = np.zeros_like(x)

        # causal 5-point derivative approximation
        for n in range(4, len(x)):
            y[n] = self.scale * (
                2 * x[n]
                + x[n - 1]
                - x[n - 3]
                - 2 * x[n - 4]
            )

        result = y.tolist()
        self.history.append(result)
        return result