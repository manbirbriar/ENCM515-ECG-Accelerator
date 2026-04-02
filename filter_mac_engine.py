from __future__ import annotations

from collections import deque

from config import OperationCycleTable
from hardware_unit import HardwareUnit, SampleToken


class FilterMacEngine(HardwareUnit):
  def __init__(self, name: str, cycle_table: OperationCycleTable, is_fixed_point: bool):
    low_pass_latency = (2 * cycle_table.shift if is_fixed_point else 2 * cycle_table.mul) + (2 * cycle_table.sub) + (2 * cycle_table.add)
    high_pass_latency = (
      (2 * cycle_table.shift if is_fixed_point else 2 * cycle_table.mul)
      + cycle_table.sub
      + (3 * cycle_table.add)
    )
    derivative_latency = (
      (3 * cycle_table.shift if is_fixed_point else 2 * cycle_table.mul)
      + cycle_table.add
      + (2 * cycle_table.sub)
    )
    super().__init__(
      name,
      latency_cycles=low_pass_latency + high_pass_latency + derivative_latency,
      initiation_interval=1,
      is_fixed_point=is_fixed_point,
    )

    zero = 0 if is_fixed_point else 0.0
    self.low_pass_x_history: deque[int | float] = deque([zero] * 12, maxlen=12)
    self.low_pass_y_1: int | float = zero
    self.low_pass_y_2: int | float = zero

    self.high_pass_x_history: deque[int | float] = deque([zero] * 32, maxlen=32)
    self.high_pass_y_1: int | float = zero

    self.derivative_x_history: deque[int | float] = deque([zero] * 4, maxlen=4)

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    x_n = int(token.value) if self.is_fixed_point else float(token.value)

    low_pass_value = self._low_pass(x_n)
    high_pass_value = self._high_pass(low_pass_value)
    derivative_value = self._derivative(high_pass_value)

    metadata = dict(token.metadata)
    metadata["low_pass"] = low_pass_value
    metadata["high_pass"] = high_pass_value
    metadata["derivative"] = derivative_value

    return SampleToken(
      sample_id=token.sample_id,
      value=derivative_value,
      ingress_cycle=token.ingress_cycle,
      metadata=metadata,
    )

  def _low_pass(self, x_n: int | float) -> int | float:
    x_n6 = self.low_pass_x_history[6]
    x_n12 = self.low_pass_x_history[0]

    if self.is_fixed_point:
      part_a = x_n - (int(x_n6) << 1) + int(x_n12)
      y_n = (int(self.low_pass_y_1) << 1) - int(self.low_pass_y_2) + part_a
    else:
      part_a = x_n - (2.0 * float(x_n6)) + float(x_n12)
      y_n = (2.0 * float(self.low_pass_y_1)) - float(self.low_pass_y_2) + part_a

    self.low_pass_y_2, self.low_pass_y_1 = self.low_pass_y_1, y_n
    self.low_pass_x_history.append(x_n)
    return y_n

  def _high_pass(self, x_n: int | float) -> int | float:
    x_n16 = self.high_pass_x_history[16]
    x_n17 = self.high_pass_x_history[15]
    x_n32 = self.high_pass_x_history[0]

    if self.is_fixed_point:
      part_a = -(int(x_n) >> 5) + int(x_n16) - int(x_n17) + (int(x_n32) >> 5)
      y_n = int(self.high_pass_y_1) + part_a
    else:
      part_a = -(float(x_n) / 32.0) + float(x_n16) - float(x_n17) + (float(x_n32) / 32.0)
      y_n = float(self.high_pass_y_1) + part_a

    self.high_pass_y_1 = y_n
    self.high_pass_x_history.append(x_n)
    return y_n

  def _derivative(self, x_n: int | float) -> int | float:
    x_n1 = self.derivative_x_history[3]
    x_n3 = self.derivative_x_history[1]
    x_n4 = self.derivative_x_history[0]

    if self.is_fixed_point:
      y_n = ((2 * int(x_n)) + int(x_n1) - int(x_n3) - (2 * int(x_n4))) >> 3
    else:
      y_n = 0.125 * ((2.0 * float(x_n)) + float(x_n1) - float(x_n3) - (2.0 * float(x_n4)))

    self.derivative_x_history.append(x_n)
    return y_n
