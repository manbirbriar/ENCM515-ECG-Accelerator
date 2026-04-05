from hardware_unit import HardwareUnit
from config import FIXED_POINT_BITS, FIXED_MUL_CYCLES, FIXED_SHIFT_CYCLES, FLOAT_MUL_CYCLES

class SquaringUnit(HardwareUnit):
  """
  Dedicated squarer: y[n] = x[n]^2
  Models a specialized multiplier with both inputs tied to the same wire.
  In fixed-point mode a right shift rescales back into Q-format after squaring.
  No history, no state — purely combinational.
  """
  def __init__(self, name: str, is_fixed_point: bool):
    # Fixed: mul + shift (Q-format rescale)
    # Float: mul only
    latency = (FIXED_MUL_CYCLES + FIXED_SHIFT_CYCLES) if is_fixed_point else FLOAT_MUL_CYCLES
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

  def compute(self, sample) -> int | float:
    if self.is_fixed_point:
      return (int(sample) * int(sample)) >> FIXED_POINT_BITS
    else:
      return sample * sample