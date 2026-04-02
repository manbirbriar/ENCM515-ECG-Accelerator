from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from clock_unit import ClockUnit
from config import (
  CLOCK_SWEEP_HZ,
  CORTEX_M4_PROFILE,
  CORTEX_M4F_PROFILE,
  CoreProfile,
  DATA_RECORDER_CAPACITY,
  DEFAULT_ACCEL_CLOCK_HZ,
  DEFAULT_CORE_PROFILE,
  FIXED_POINT_SCALE,
  INGRESS_FIFO_DEPTH,
  MWI_WINDOW_SIZE,
  REFRACTORY_PERIOD_SAMPLES,
  SAMPLE_RATE_HZ,
  get_cycle_table,
)
from control_unit import ControlUnit
from data_recorder import DataRecorder
from data_uploader import DataUploader
from filter_mac_engine import FilterMacEngine
from mwi_unit import MWIUnit
from sample_queue import InputBuffer
from squaring_unit import SquaringUnit
from threshold_unit import PeakDetectorUnit


def _parse_signal_line(line: str) -> tuple[int, int]:
  parts = line.split()
  gain = int(parts[2].split("/")[0])
  baseline = int(parts[4])
  return gain, baseline


def load_data(record_path: str, channel: int = 0) -> np.ndarray:
  record_root = Path(record_path)
  header_path = record_root.with_suffix(".hea")
  data_path = record_root.with_suffix(".dat")

  header_lines = header_path.read_text().splitlines()
  header_fields = header_lines[0].split()
  signal_count = int(header_fields[1])
  signal_lines = header_lines[1:1 + signal_count]
  gains = []
  baselines = []
  for line in signal_lines:
    gain, baseline = _parse_signal_line(line)
    gains.append(gain)
    baselines.append(baseline)

  raw = np.frombuffer(data_path.read_bytes(), dtype=np.uint8)
  if len(raw) % 3 != 0:
    raise ValueError(f"Unexpected MIT-BIH record length in {data_path}")

  packed = raw.reshape(-1, 3)
  channel_0 = ((packed[:, 1] & 0x0F) << 8) | packed[:, 0]
  channel_1 = (packed[:, 1] << 4) | packed[:, 2]

  channel_0 = channel_0.astype(np.int16)
  channel_1 = channel_1.astype(np.int16)
  channel_0[channel_0 >= 2048] -= 4096
  channel_1[channel_1 >= 2048] -= 4096

  adc = np.column_stack((channel_0, channel_1))
  physical = (adc[:, channel] - baselines[channel]) / gains[channel]
  return physical.astype(np.float32)


def _pipeline_drained(units: list[Any]) -> bool:
  for unit in units:
    if hasattr(unit, "inflight") and unit.inflight:
      return False
    if hasattr(unit, "output_buffer") and unit.output_buffer:
      return False
  return True


def run_simulation(
  float_samples: np.ndarray,
  is_fixed: bool,
  core_profile: CoreProfile = DEFAULT_CORE_PROFILE,
  accel_clock_hz: int = DEFAULT_ACCEL_CLOCK_HZ,
  sample_rate_hz: int = SAMPLE_RATE_HZ,
  ingress_fifo_depth: int = INGRESS_FIFO_DEPTH,
) -> dict[str, Any]:
  cycle_table = get_cycle_table(core_profile, is_fixed_point=is_fixed)
  input_samples = (float_samples * FIXED_POINT_SCALE).astype(np.int32) if is_fixed else float_samples.astype(np.float32)

  recorders = {
    "ingress": DataRecorder("ingress", DATA_RECORDER_CAPACITY),
    "filter_mac": DataRecorder("filter_mac", DATA_RECORDER_CAPACITY),
    "squaring": DataRecorder("squaring", DATA_RECORDER_CAPACITY),
    "mwi": DataRecorder("mwi", DATA_RECORDER_CAPACITY),
    "peak_detector": DataRecorder("peak_detector", DATA_RECORDER_CAPACITY),
  }

  input_buffer = InputBuffer("input_buffer", ingress_fifo_depth)
  sample_ingress = DataUploader("sample_ingress", input_samples, sample_rate_hz=sample_rate_hz, accel_clock_hz=accel_clock_hz)
  sample_ingress.attach_recorder(recorders["ingress"])

  filter_mac = FilterMacEngine("filter_mac_engine", cycle_table=cycle_table, is_fixed_point=is_fixed)
  squaring = SquaringUnit("squaring_unit", cycle_table=cycle_table, is_fixed_point=is_fixed)
  mwi = MWIUnit("moving_window_integrator", cycle_table=cycle_table, mwi_window_size=MWI_WINDOW_SIZE, is_fixed_point=is_fixed)
  peak_detector = PeakDetectorUnit(
    "peak_detector",
    cycle_table=cycle_table,
    sample_rate=sample_rate_hz,
    refractory_period_samples=REFRACTORY_PERIOD_SAMPLES,
    is_fixed_point=is_fixed,
  )

  filter_mac.attach_recorder(recorders["filter_mac"])
  squaring.attach_recorder(recorders["squaring"])
  mwi.attach_recorder(recorders["mwi"])
  peak_detector.attach_recorder(recorders["peak_detector"])

  sample_ingress.connect(input_buffer)
  input_buffer.connect(filter_mac)
  filter_mac.connect(squaring).connect(mwi).connect(peak_detector)

  pipeline_units = [filter_mac, squaring, mwi, peak_detector]
  clock_units = [sample_ingress, input_buffer, *pipeline_units]
  clock = ClockUnit().subscribe_many(clock_units)

  while True:
    if input_buffer.is_empty() and _pipeline_drained(pipeline_units) and not sample_ingress.is_done():
      next_arrival_cycle = sample_ingress.next_arrival_cycle()
      if next_arrival_cycle is not None and next_arrival_cycle > clock.cycle + 1:
        skipped_cycles = next_arrival_cycle - clock.cycle - 1
        input_buffer.idle_cycles += skipped_cycles
        for unit in pipeline_units:
          unit.idle_cycles += skipped_cycles
        clock.cycle = next_arrival_cycle - 1

    clock.tick()
    if sample_ingress.is_done() and input_buffer.is_empty() and _pipeline_drained(pipeline_units):
      break

  control = ControlUnit(sample_rate_hz=sample_rate_hz, accel_clock_hz=accel_clock_hz, core_profile=core_profile)
  metrics = control.summarize(
    total_cycles=clock.cycle,
    accepted_samples=sample_ingress.accepted_count,
    attempted_samples=sample_ingress.sample_index,
    dropped_samples=sample_ingress.dropped_samples,
    input_buffer=input_buffer,
    stages=pipeline_units,
    peak_samples=peak_detector.peaks,
  )
  metrics["average_end_to_end_latency_cycles"] = (
    peak_detector.latency_sum / peak_detector.latency_count if peak_detector.latency_count else 0.0
  )
  metrics["max_end_to_end_latency_cycles"] = peak_detector.max_latency

  return {
    "recorders": recorders,
    "peak_detector_unit": peak_detector,
    "metrics": metrics,
    "clock": clock,
    "input_buffer": input_buffer,
  }


def build_stage_models(core_profile: CoreProfile, is_fixed_for_run: bool) -> list[Any]:
  cycle_table = get_cycle_table(core_profile, is_fixed_point=is_fixed_for_run)
  return [
    FilterMacEngine("filter_mac_engine", cycle_table=cycle_table, is_fixed_point=is_fixed_for_run),
    SquaringUnit("squaring_unit", cycle_table=cycle_table, is_fixed_point=is_fixed_for_run),
    MWIUnit("moving_window_integrator", cycle_table=cycle_table, mwi_window_size=MWI_WINDOW_SIZE, is_fixed_point=is_fixed_for_run),
    PeakDetectorUnit(
      "peak_detector",
      cycle_table=cycle_table,
      sample_rate=SAMPLE_RATE_HZ,
      refractory_period_samples=REFRACTORY_PERIOD_SAMPLES,
      is_fixed_point=is_fixed_for_run,
    ),
  ]


def print_cycle_summary(core_profile: CoreProfile, is_fixed_for_run: bool) -> None:
  stages = build_stage_models(core_profile, is_fixed_for_run)
  mode_label = "Fixed" if is_fixed_for_run else "Float"
  total_latency = sum(stage.latency_cycles for stage in stages)
  print(f"\n{core_profile.label} {mode_label} per-sample stage summary:")
  for stage in stages:
    print(f"  {stage.name}: latency={stage.latency_cycles} ii={stage.initiation_interval}")
  print(f"  total_pipeline_latency_cycles: {total_latency}")


def compare_recorders(float_recorders: dict[str, DataRecorder], fixed_recorders: dict[str, DataRecorder]) -> dict[str, float]:
  rmse_by_stage: dict[str, float] = {}
  for stage_name, float_recorder in float_recorders.items():
    fixed_signal = np.asarray(fixed_recorders[stage_name].get_signal(), dtype=np.float64)
    float_signal = np.asarray(float_recorder.get_signal(), dtype=np.float64)
    if stage_name != "peak_detector":
      fixed_signal = fixed_signal / FIXED_POINT_SCALE

    min_len = min(len(float_signal), len(fixed_signal))
    if min_len == 0:
      rmse_by_stage[stage_name] = 0.0
      continue

    error = float_signal[:min_len] - fixed_signal[:min_len]
    rmse_by_stage[stage_name] = float(np.sqrt(np.mean(error * error)))
  return rmse_by_stage


def print_metrics(label: str, metrics: dict[str, Any], bpm: float) -> None:
  print(f"\n{label} streaming accelerator metrics:")
  print(f"  core_profile: {metrics['core_profile']}")
  print(f"  accel_clock_hz: {metrics['accel_clock_hz']}")
  print(f"  total_cycles: {metrics['total_cycles']}")
  print(f"  accepted_samples: {metrics['accepted_samples']}")
  print(f"  attempted_samples: {metrics['attempted_samples']}")
  print(f"  dropped_samples: {metrics['dropped_samples']}")
  print(f"  effective_samples_per_second: {metrics['effective_samples_per_second']:.2f}")
  print(f"  throughput_margin_vs_input: {metrics['throughput_margin_vs_input']:.2f}x")
  print(f"  sustainable_samples_per_second: {metrics['sustainable_samples_per_second']:.2f}")
  print(f"  sustainable_throughput_margin_vs_input: {metrics['sustainable_throughput_margin_vs_input']:.2f}x")
  print(f"  input_buffer_max_occupancy: {metrics['input_buffer_max_occupancy']}")
  print(f"  input_buffer_overflows: {metrics['input_buffer_overflows']}")
  print(f"  front_end_idle_cycles: {metrics['front_end_idle_cycles']}")
  print(f"  detected_peak_count: {metrics['detected_peak_count']}")
  print(f"  estimated_bpm: {bpm:.2f}")
  print(f"  avg_end_to_end_latency_cycles: {metrics['average_end_to_end_latency_cycles']:.2f}")
  print(f"  max_end_to_end_latency_cycles: {metrics['max_end_to_end_latency_cycles']}")
  print(f"  energy_per_cycle_j: {metrics['energy_per_cycle_j']:.3e}")
  print(f"  total_energy_j: {metrics['total_energy_j']:.3e}")
  print(f"  average_power_w: {metrics['average_power_w']:.3e}")

  print("  stage_metrics:")
  for stage in metrics["stage_metrics"]:
    print(
      f"    {stage.name}: util={stage.utilization:.3f} "
      f"idle={stage.idle_cycles} stalled={stage.stalled_cycles} "
      f"downstream_stall={stage.downstream_stall_cycles}"
    )


def run_frequency_sweep(
  float_samples: np.ndarray,
  core_profile: CoreProfile,
  is_fixed: bool,
  clock_values_hz: tuple[int, ...] = CLOCK_SWEEP_HZ,
) -> list[dict[str, Any]]:
  sweep_results = []
  for clock_hz in clock_values_hz:
    results = run_simulation(float_samples, is_fixed=is_fixed, core_profile=core_profile, accel_clock_hz=clock_hz)
    sweep_results.append(
      {
        "clock_hz": clock_hz,
        "core_profile": core_profile.label,
        "mode": "fixed" if is_fixed else "float",
        "accepted_samples": results["metrics"]["accepted_samples"],
        "dropped_samples": results["metrics"]["dropped_samples"],
        "front_end_idle_cycles": results["metrics"]["front_end_idle_cycles"],
        "throughput_margin": results["metrics"]["sustainable_throughput_margin_vs_input"],
        "total_energy_j": results["metrics"]["total_energy_j"],
        "average_power_w": results["metrics"]["average_power_w"],
      }
    )
  return sweep_results


def print_frequency_sweep(sweep_results: list[dict[str, Any]]) -> None:
  print("\nFrequency sweep:")
  for entry in sweep_results:
    print(
      f"  {entry['core_profile']} {entry['mode']} @ {entry['clock_hz']} Hz: "
      f"accepted={entry['accepted_samples']} dropped={entry['dropped_samples']} "
      f"idle_cycles={entry['front_end_idle_cycles']} throughput_margin={entry['throughput_margin']:.2f}x "
      f"power={entry['average_power_w']:.3e} W"
    )


def plot_data_recorders(recorders: dict[str, DataRecorder], patient_number: int) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError("matplotlib is required for plotting") from exc

  ordered_names = ["ingress", "filter_mac", "squaring", "mwi", "peak_detector"]
  fig, axes = plt.subplots(len(ordered_names), 1, figsize=(12, 14), sharex=True)

  for axis, name in zip(axes, ordered_names):
    signal = recorders[name].get_signal()
    axis.plot(np.arange(len(signal)), signal)
    axis.set_title(name)
    axis.grid(True, alpha=0.3)

  axes[-1].set_xlabel("Sample")
  plt.suptitle(f"Patient {patient_number} Streaming ECG Accelerator", fontsize=20)
  plt.tight_layout()
  plt.show()


__all__ = [
  "CORTEX_M4_PROFILE",
  "CORTEX_M4F_PROFILE",
  "compare_recorders",
  "load_data",
  "plot_data_recorders",
  "print_cycle_summary",
  "print_frequency_sweep",
  "print_metrics",
  "run_frequency_sweep",
  "run_simulation",
]
