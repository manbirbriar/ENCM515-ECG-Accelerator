from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import CoreProfile, REFERENCE_DEVICE_NAME, REFERENCE_ENERGY_PER_CYCLE_J, REFERENCE_RUN_CURRENT_UA_PER_MHZ, REFERENCE_SUPPLY_VOLTAGE_V
from hardware_unit import HardwareUnit
from sample_queue import InputBuffer


@dataclass(slots=True)
class StageMetrics:
  name: str
  latency_cycles: int
  initiation_interval: int
  accepted_samples: int
  completed_samples: int
  emitted_samples: int
  utilization: float
  idle_cycles: int
  stalled_cycles: int
  downstream_stall_cycles: int


class ControlUnit:
  def __init__(self, sample_rate_hz: int, accel_clock_hz: int, core_profile: CoreProfile):
    self.sample_rate_hz = sample_rate_hz
    self.accel_clock_hz = accel_clock_hz
    self.core_profile = core_profile

  def summarize(
    self,
    total_cycles: int,
    accepted_samples: int,
    attempted_samples: int,
    dropped_samples: int,
    input_buffer: InputBuffer,
    stages: list[HardwareUnit],
    peak_samples: list[int],
  ) -> dict[str, Any]:
    throughput_sps = 0.0
    if total_cycles > 0:
      throughput_sps = accepted_samples * self.accel_clock_hz / total_cycles

    max_stage_ii = max((stage.initiation_interval for stage in stages), default=1)
    sustainable_samples_per_second = self.accel_clock_hz / max_stage_ii

    stage_metrics = [
      StageMetrics(
        name=stage.name,
        latency_cycles=stage.latency_cycles,
        initiation_interval=stage.initiation_interval,
        accepted_samples=stage.accepted_count,
        completed_samples=stage.completed_count,
        emitted_samples=stage.emitted_count,
        utilization=stage.utilization(total_cycles),
        idle_cycles=stage.idle_cycles,
        stalled_cycles=stage.stalled_cycles,
        downstream_stall_cycles=stage.downstream_stall_cycles,
      )
      for stage in stages
    ]

    total_energy_j = total_cycles * REFERENCE_ENERGY_PER_CYCLE_J
    average_power_w = total_energy_j / (total_cycles / self.accel_clock_hz) if total_cycles > 0 else 0.0

    return {
      "core_profile": self.core_profile.label,
      "core_profile_notes": self.core_profile.notes,
      "sample_rate_hz": self.sample_rate_hz,
      "accel_clock_hz": self.accel_clock_hz,
      "total_cycles": total_cycles,
      "accepted_samples": accepted_samples,
      "attempted_samples": attempted_samples,
      "dropped_samples": dropped_samples,
      "effective_samples_per_second": throughput_sps,
      "throughput_margin_vs_input": throughput_sps / self.sample_rate_hz if self.sample_rate_hz else 0.0,
      "sustainable_samples_per_second": sustainable_samples_per_second,
      "sustainable_throughput_margin_vs_input": (
        sustainable_samples_per_second / self.sample_rate_hz if self.sample_rate_hz else 0.0
      ),
      "input_buffer_max_occupancy": input_buffer.max_occupancy,
      "input_buffer_overflows": input_buffer.overflow_count,
      "front_end_idle_cycles": stages[0].idle_cycles if stages else 0,
      "detected_peak_count": len(peak_samples),
      "detected_peak_samples": peak_samples,
      "power_reference_device": REFERENCE_DEVICE_NAME,
      "power_reference_voltage_v": REFERENCE_SUPPLY_VOLTAGE_V,
      "power_reference_current_ua_per_mhz": REFERENCE_RUN_CURRENT_UA_PER_MHZ,
      "energy_per_cycle_j": REFERENCE_ENERGY_PER_CYCLE_J,
      "total_energy_j": total_energy_j,
      "average_power_w": average_power_w,
      "stage_metrics": stage_metrics,
    }
