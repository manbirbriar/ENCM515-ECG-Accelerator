from __future__ import annotations

from collections import deque

from config import FIXED_POINT_SCALE, OperationCycleTable
from hardware_unit import HardwareUnit, SampleToken


class PeakDetectorUnit(HardwareUnit):
  def __init__(
    self,
    name: str,
    cycle_table: OperationCycleTable,
    sample_rate: int,
    refractory_period_samples: int,
    is_fixed_point: bool,
  ):
    peak_detect_latency = 3 * cycle_table.compare
    gate_latency = cycle_table.compare + cycle_table.sub + cycle_table.compare
    adaptive_update_latency = (2 * cycle_table.mul) + cycle_table.add
    threshold_update_latency = cycle_table.sub + cycle_table.mul + cycle_table.add
    super().__init__(
      name,
      latency_cycles=peak_detect_latency + gate_latency + adaptive_update_latency + threshold_update_latency,
      initiation_interval=1,
      is_fixed_point=is_fixed_point,
    )
    self.sample_rate = sample_rate
    self.refractory_period_samples = refractory_period_samples

    if is_fixed_point:
      self.spki: int | float = 2 * FIXED_POINT_SCALE
      self.npki: int | float = FIXED_POINT_SCALE // 2
      self.threshold: int | float = FIXED_POINT_SCALE
      zero = 0
    else:
      self.spki = 2.0
      self.npki = 0.5
      self.threshold = 1.0
      zero = 0.0

    self.last_peak_sample = -refractory_period_samples
    self.peaks: list[int] = []
    self.peak_values: list[int | float] = []
    self.recent_values: deque[int | float] = deque([zero, zero], maxlen=2)

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    current_value = int(token.value) if self.is_fixed_point else float(token.value)
    previous_value = self.recent_values[-1]
    two_back_value = self.recent_values[0]
    pulse_value = 0 if self.is_fixed_point else 0.0

    is_local_peak = previous_value > two_back_value and previous_value >= current_value
    candidate_sample = token.sample_id - 1

    if is_local_peak and candidate_sample >= 0:
      outside_refractory = (candidate_sample - self.last_peak_sample) > self.refractory_period_samples
      if previous_value > self.threshold and outside_refractory:
        self.peaks.append(candidate_sample)
        self.peak_values.append(previous_value)
        self.last_peak_sample = candidate_sample
        if self.is_fixed_point:
          self.spki = (int(previous_value) + (7 * int(self.spki))) >> 3
        else:
          self.spki = (0.125 * previous_value) + (0.875 * self.spki)
        pulse_value = 1 if self.is_fixed_point else 1.0
      else:
        if self.is_fixed_point:
          self.npki = (int(previous_value) + (7 * int(self.npki))) >> 3
        else:
          self.npki = (0.125 * previous_value) + (0.875 * self.npki)
    else:
      if self.is_fixed_point:
        self.npki = (int(current_value) + (7 * int(self.npki))) >> 3
      else:
        self.npki = (0.125 * current_value) + (0.875 * self.npki)

    if self.is_fixed_point:
      self.threshold = int(self.npki) + ((int(self.spki) - int(self.npki)) >> 2)
    else:
      self.threshold = self.npki + (0.25 * (self.spki - self.npki))
    self.recent_values.append(current_value)
    return SampleToken(sample_id=token.sample_id, value=pulse_value, ingress_cycle=token.ingress_cycle, metadata=token.metadata)

  def get_bpm(self) -> float:
    if len(self.peaks) < 2:
      return 0.0

    rr_intervals = [current - previous for previous, current in zip(self.peaks, self.peaks[1:])]
    average_rr = sum(rr_intervals) / len(rr_intervals)
    return (60.0 * self.sample_rate) / average_rr


ThresholdUnit = PeakDetectorUnit
