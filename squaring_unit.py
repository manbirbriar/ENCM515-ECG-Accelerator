from __future__ import annotations

from config import FIXED_POINT_BITS, OperationCycleTable
from hardware_unit import HardwareUnit, SampleToken


class SquaringUnit(HardwareUnit):
  def __init__(self, name: str, cycle_table: OperationCycleTable, is_fixed_point: bool):
    latency = (cycle_table.mul + cycle_table.shift) if is_fixed_point else cycle_table.mul
    super().__init__(name, latency_cycles=latency, initiation_interval=1, is_fixed_point=is_fixed_point)

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    if self.is_fixed_point:
      value = int(token.value)
      y_n = (value * value) >> FIXED_POINT_BITS
    else:
      value = float(token.value)
      y_n = value * value

    return SampleToken(sample_id=token.sample_id, value=y_n, ingress_cycle=token.ingress_cycle, metadata=token.metadata)
