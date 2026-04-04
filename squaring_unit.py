from hardware_unit import HardwareUnit
from config import FIXED_POINT_BITS, FIXED_MUL_CYCLES, FIXED_SHIFT_CYCLES, FLOAT_MUL_CYCLES

class SquaringUnit(HardwareUnit):
  """
  Dedicated squarer: y[n] = x[n]^2
  Both multiplier inputs tied to the same wire.
  Fixed-point includes a right shift for Q-format rescaling.
  No state, no history — purely combinational.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    latency = (FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES) if is_fixed_point else FLOAT_MUL_CYCLES
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

  def compute(self, sample) -> int | float:
    if self.is_fixed_point:
      return (int(sample) * int(sample)) >> FIXED_POINT_BITS
    else:
      return sample * sample
