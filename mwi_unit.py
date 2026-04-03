import numpy as np

from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer
from config import (
  MWI_WINDOW_SIZE,
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_SHIFT_CYCLES, FIXED_MUL_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES,
)

# Magic number approximation for fixed-point division by MWI_WINDOW_SIZE (54)
# y = (x * MWI_NORM_MAGIC) >> MWI_NORM_SHIFT  ≈  x / 54
# Error: 0.0099% vs true division
# Intermediate product fits in int64 (max ~8.6B, well under 2^63)
# This is how real DSPs handle non-power-of-2 division — avoids expensive hardware divider
MWI_NORM_MAGIC: int = 4855   # round(2^18 / 54)
MWI_NORM_SHIFT: int = 18

class MWIUnit(HardwareUnit):
  """
  Moving Window Integrator: y[n] = (1/N) * sum(x[n-N+1] ... x[n])

  Implemented as a running sum with a circular buffer:
    running_sum += x[n] - x[n-N]
    y[n] = running_sum / N

  Fixed-point normalization uses a multiply-shift approximation:
    y[n] = (running_sum * 4855) >> 18  ≈  running_sum / 54
  This models how a real DSP implements non-power-of-2 division
  without a hardware divider.

  Hardware: adder + subtractor + accumulator register + N-sample circular buffer (SRAM).
  The dominant hardware cost is the 54-sample buffer, not the arithmetic.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    # Fixed: sub + add + mul + shift (multiply-shift normalization)
    # Float: sub + add + mul (multiply by norm_constant)
    latency = (
      (FIXED_SUB_CYCLES + FIXED_ADD_CYCLES + FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES) if is_fixed_point
      else (FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES + FLOAT_MUL_CYCLES)
    )
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.mwi_window_size = MWI_WINDOW_SIZE
    self.norm_constant = 1.0 / MWI_WINDOW_SIZE

    # 54-sample circular buffer — this IS the algorithm, not just history
    dtype = int if is_fixed_point else float
    self.window_buffer = CircularBuffer(MWI_WINDOW_SIZE, dtype=dtype)
    self.running_sum = 0

  def compute(self, sample) -> int | float:
    # Oldest sample leaving the window
    oldest = self.window_buffer[-self.mwi_window_size]

    # Update running sum: add newest, drop oldest
    self.running_sum += sample - oldest

    # Push new sample into window
    self.window_buffer.push(sample)

    if self.is_fixed_point:
      # Multiply-shift approximation of division by 54
      # Uses int64 to prevent overflow during intermediate multiplication
      return int((np.int64(self.running_sum) * MWI_NORM_MAGIC) >> MWI_NORM_SHIFT)
    else:
      return self.running_sum * self.norm_constant