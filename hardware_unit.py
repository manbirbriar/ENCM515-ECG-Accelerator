from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from data_recorder import DataRecorder


@dataclass(slots=True)
class SampleToken:
  sample_id: int
  value: Any
  ingress_cycle: int
  metadata: dict[str, Any] = field(default_factory=dict)


class HardwareUnit(ABC):
  def __init__(
    self,
    name: str,
    latency_cycles: int = 1,
    initiation_interval: int = 1,
    is_fixed_point: bool = False,
    output_buffer_capacity: int = 1,
  ):
    self.name = name
    self.latency_cycles = max(1, latency_cycles)
    self.initiation_interval = max(1, initiation_interval)
    self.is_fixed_point = is_fixed_point

    self.next_unit: HardwareUnit | None = None
    self.recorder: DataRecorder | None = None

    self.next_issue_cycle = 1
    self.output_buffer_capacity = max(1, output_buffer_capacity)
    self.pipeline_capacity = max(1, ceil(self.latency_cycles / self.initiation_interval))
    self.inflight: deque[tuple[int, SampleToken]] = deque()
    self.output_buffer: deque[SampleToken] = deque()

    self.accepted_count = 0
    self.completed_count = 0
    self.emitted_count = 0
    self.busy_cycles = 0
    self.idle_cycles = 0
    self.stalled_cycles = 0
    self.input_stall_cycles = 0
    self.downstream_stall_cycles = 0
    self.latency_count = 0
    self.latency_sum = 0
    self.max_latency = 0

  def connect(self, next_unit: HardwareUnit) -> HardwareUnit:
    self.next_unit = next_unit
    return next_unit

  def attach_recorder(self, recorder: DataRecorder) -> None:
    self.recorder = recorder

  def can_accept(self, current_cycle: int) -> bool:
    issue_ready = current_cycle >= self.next_issue_cycle
    capacity_ready = len(self.inflight) < self.pipeline_capacity
    return issue_ready and capacity_ready

  def accept(self, token: SampleToken, current_cycle: int) -> bool:
    if not self.can_accept(current_cycle):
      self.input_stall_cycles += 1
      return False

    output_token = self.process_token(token, current_cycle)
    ready_cycle = current_cycle + self.latency_cycles
    self.inflight.append((ready_cycle, output_token))
    self.next_issue_cycle = current_cycle + self.initiation_interval
    self.accepted_count += 1
    return True

  def tick(self, current_cycle: int) -> None:
    active = bool(self.inflight or self.output_buffer)
    if active:
      self.busy_cycles += 1
    else:
      self.idle_cycles += 1

    if self.output_buffer:
      forwarded = self._forward_one(current_cycle)
      if not forwarded:
        self.downstream_stall_cycles += 1

    if self.inflight and len(self.output_buffer) >= self.output_buffer_capacity:
      next_ready_cycle = self.inflight[0][0]
      if next_ready_cycle <= current_cycle:
        self.stalled_cycles += 1
        return

    while self.inflight and self.inflight[0][0] <= current_cycle:
      if len(self.output_buffer) >= self.output_buffer_capacity:
        self.stalled_cycles += 1
        break

      _, output_token = self.inflight.popleft()
      self.output_buffer.append(output_token)
      self.completed_count += 1
      token_latency = current_cycle - output_token.ingress_cycle
      self.latency_count += 1
      self.latency_sum += token_latency
      self.max_latency = max(self.max_latency, token_latency)
      if self.recorder is not None:
        self.recorder.record_token(output_token, current_cycle)

      if self.output_buffer:
        self._forward_one(current_cycle)
        break

  def _forward_one(self, current_cycle: int) -> bool:
    if not self.output_buffer:
      return False

    if self.next_unit is None:
      self.output_buffer.popleft()
      self.emitted_count += 1
      return True

    if self.next_unit.can_accept(current_cycle):
      token = self.output_buffer.popleft()
      self.next_unit.accept(token, current_cycle)
      self.emitted_count += 1
      return True

    return False

  @abstractmethod
  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    raise NotImplementedError

  def is_stalled(self) -> bool:
    return bool(self.output_buffer) and self.next_unit is not None

  def utilization(self, total_cycles: int) -> float:
    if total_cycles <= 0:
      return 0.0
    return self.busy_cycles / total_cycles

  def __repr__(self) -> str:
    return (
      f"<{self.__class__.__name__} name={self.name} latency={self.latency_cycles} "
      f"ii={self.initiation_interval} inflight={len(self.inflight)} "
      f"output_buffer={len(self.output_buffer)}>"
    )
