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
# Intermediate product requires int64 (moves to wide accumulator register in hardware)
MWI_NORM_MAGIC: int = 4855   # round(2^18 / 54)
MWI_NORM_SHIFT: int = 18

class MWIUnit(HardwareUnit):
  """
  Moving Window Integrator: y[n] = (1/N) * sum(x[n-N+1] ... x[n])

  Running sum implementation:
    running_sum += x[n] - x[n-N]
    y[n] = running_sum / N

  Fixed-point normalization uses multiply-shift approximation of /54.
  Intermediate product promoted to int64 (wide accumulator register in hardware)
  then result moved back to int32.

  Hardware: adder + subtractor + accumulator register + 54-sample SRAM.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    latency = (
      (FIXED_SUB_CYCLES + FIXED_ADD_CYCLES + FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES) if is_fixed_point
      else (FLOAT_SUB_CYCLES + FLOAT_ADD_CYCLES + FLOAT_MUL_CYCLES)
    )
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.mwi_window_size = MWI_WINDOW_SIZE
    self.norm_constant = 1.0 / MWI_WINDOW_SIZE

    dtype = int if is_fixed_point else float
    self.window_buffer = CircularBuffer(MWI_WINDOW_SIZE, dtype=dtype)
    self.running_sum = 0

  def compute(self, sample) -> int | float:
    oldest = self.window_buffer[-self.mwi_window_size]
    self.running_sum += sample - oldest
    self.window_buffer.push(sample)

    if self.is_fixed_point:
      # Multiply-shift approximation of /54
      # Promote to int64 (wide accumulator), shift back to int32
      return int((np.int64(self.running_sum) * MWI_NORM_MAGIC) >> MWI_NORM_SHIFT)
    else:
      return self.running_sum * self.norm_constant
