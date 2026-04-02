from __future__ import annotations

from collections import deque

from config import OperationCycleTable
from hardware_unit import HardwareUnit, SampleToken


class MWIUnit(HardwareUnit):
  def __init__(self, name: str, cycle_table: OperationCycleTable, mwi_window_size: int, is_fixed_point: bool):
    latency = cycle_table.sub + cycle_table.add + cycle_table.mul
    super().__init__(name, latency_cycles=latency, initiation_interval=1, is_fixed_point=is_fixed_point)

    zero = 0 if is_fixed_point else 0.0
    self.mwi_window_size = mwi_window_size
    self.window: deque[int | float] = deque([zero] * mwi_window_size, maxlen=mwi_window_size)
    self.running_sum: int | float = zero

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    current_value = int(token.value) if self.is_fixed_point else float(token.value)
    expired_value = self.window[0]
    self.running_sum += current_value - expired_value
    self.window.append(current_value)

    if self.is_fixed_point:
      y_n = int(self.running_sum) // self.mwi_window_size
    else:
      y_n = float(self.running_sum) / self.mwi_window_size

    return SampleToken(sample_id=token.sample_id, value=y_n, ingress_cycle=token.ingress_cycle, metadata=token.metadata)
