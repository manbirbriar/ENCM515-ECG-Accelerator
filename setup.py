from __future__ import annotations

from accelerator_sim import (
  CORTEX_M4_PROFILE,
  CORTEX_M4F_PROFILE,
  compare_recorders,
  load_data,
  plot_data_recorders,
  print_cycle_summary,
  print_frequency_sweep,
  print_metrics,
  run_frequency_sweep,
  run_simulation,
)
from config import CLOCK_SWEEP_HZ


def print_core_mode_comparison(comparisons: list[tuple[str, dict, float]]) -> None:
  print("\nCore/mode comparison @ 1 MHz")
  for label, results, bpm in comparisons:
    metrics = results["metrics"]
    print(
      f"  {label}: "
      f"latency={metrics['average_end_to_end_latency_cycles']:.2f} cycles, "
      f"peaks={metrics['detected_peak_count']}, "
      f"bpm={bpm:.2f}, "
      f"drops={metrics['dropped_samples']}, "
      f"power={metrics['average_power_w']:.3e} W"
    )


def print_rmse_against_reference(
  reference_label: str,
  reference_recorders: dict,
  comparisons: list[tuple[str, dict]],
) -> None:
  print(f"\nRMSE vs {reference_label}")
  for label, results in comparisons:
    rmse_by_stage = compare_recorders(reference_recorders, results["recorders"])
    print(f"  {label}:")
    for stage_name, rmse in rmse_by_stage.items():
      print(f"    {stage_name}: {rmse:.8f}")


if __name__ == "__main__":
  patient_number = 116
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"
  analysis_samples = load_data(record_path)[:5000]

  print_cycle_summary(CORTEX_M4_PROFILE, is_fixed_for_run=True)
  m4_fixed_results = run_simulation(analysis_samples, is_fixed=True, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=CLOCK_SWEEP_HZ[1])
  m4_fixed_bpm = m4_fixed_results["peak_detector_unit"].get_bpm()
  print_metrics("Cortex-M4 Fixed @ 1 MHz", m4_fixed_results["metrics"], m4_fixed_bpm)

  print_cycle_summary(CORTEX_M4_PROFILE, is_fixed_for_run=False)
  m4_soft_float_results = run_simulation(analysis_samples, is_fixed=False, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=CLOCK_SWEEP_HZ[1])
  m4_soft_float_bpm = m4_soft_float_results["peak_detector_unit"].get_bpm()
  print_metrics("Cortex-M4 Soft-Float @ 1 MHz", m4_soft_float_results["metrics"], m4_soft_float_bpm)

  print_cycle_summary(CORTEX_M4F_PROFILE, is_fixed_for_run=False)
  m4f_float_results = run_simulation(analysis_samples, is_fixed=False, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=CLOCK_SWEEP_HZ[1])
  m4f_float_bpm = m4f_float_results["peak_detector_unit"].get_bpm()
  print_metrics("Cortex-M4F Float @ 1 MHz", m4f_float_results["metrics"], m4f_float_bpm)

  print_core_mode_comparison(
    [
      ("Cortex-M4 Fixed", m4_fixed_results, m4_fixed_bpm),
      ("Cortex-M4 Soft-Float", m4_soft_float_results, m4_soft_float_bpm),
      ("Cortex-M4F Hardware-Float", m4f_float_results, m4f_float_bpm),
    ]
  )

  print("\nCortex-M4 fixed frequency comparison")
  print_frequency_sweep(run_frequency_sweep(analysis_samples, core_profile=CORTEX_M4_PROFILE, is_fixed=True))

  print("\nCortex-M4 soft-float frequency comparison")
  print_frequency_sweep(run_frequency_sweep(analysis_samples, core_profile=CORTEX_M4_PROFILE, is_fixed=False))

  print("\nCortex-M4F float frequency comparison")
  print_frequency_sweep(run_frequency_sweep(analysis_samples, core_profile=CORTEX_M4F_PROFILE, is_fixed=False))

  print_rmse_against_reference(
    "Cortex-M4F Hardware-Float",
    m4f_float_results["recorders"],
    [
      ("Cortex-M4 Fixed", m4_fixed_results),
      ("Cortex-M4 Soft-Float", m4_soft_float_results),
    ],
  )

  try:
    plot_data_recorders(m4f_float_results["recorders"], patient_number=patient_number)
    plot_data_recorders(m4_fixed_results["recorders"], patient_number=patient_number)
  except RuntimeError as error:
    print(f"\nPlotting skipped: {error}")
