from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer
from config import (
  MWI_WINDOW_SIZE,
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_SHIFT_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES,
)

class MWIUnit(HardwareUnit):
  """
  Moving Window Integrator: y[n] = (1/N) * sum(x[n-N+1] ... x[n])
  
  Implemented as a running sum with a circular buffer:
    running_sum += x[n] - x[n-N]
    y[n] = running_sum / N

  Hardware: adder + subtractor + accumulator register + N-sample circular buffer (SRAM).
  The dominant hardware cost is the 54-sample buffer, not the arithmetic.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    # 1 sub (drop oldest) + 1 add (running sum update) + 1 shift/mul (normalize)
    latency = (
      (FIXED_SUB_CYCLES + FIXED_ADD_CYCLES + FIXED_SHIFT_CYCLES) if is_fixed_point
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

    # Update running sum: add new, drop oldest
    self.running_sum += sample - oldest

    # Push new sample into window
    self.window_buffer.push(sample)

    if self.is_fixed_point:
      return int(self.running_sum) // self.mwi_window_size
    else:
      return self.running_sum * self.norm_constant