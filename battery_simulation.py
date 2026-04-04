import numpy as np

from config import FIXED_POINT_SCALE, SAMPLE_RATE
from simulator import get_patient_bpm, load_data, run_simulation


ENERGY_PER_CYCLE_PJ = 12.26
SWEEP_FREQUENCIES_HZ = [1000, 10_000, 25_000, 50_000, 100_000, 200_000]


def aggregate_cycle_counters(units: list):
  busy = idle = stalled = 0
  for unit in units:
    if hasattr(unit, "busy_cycles"):
      busy += unit.busy_cycles
      idle += unit.idle_cycles
      stalled += unit.stalled_cycles
  total = busy + idle + stalled
  util = busy / total if total > 0 else 0.0
  return busy, idle, stalled, util


def run_frequency_sweep(float_lead0, float_lead1, fixed_lead0, fixed_lead1):
  empty_float_lead = np.array([], dtype=np.float32)
  empty_fixed_lead = np.array([], dtype=np.int32)
  modes = [
    ("Float", False, float_lead0, empty_float_lead),
    ("Fixed", True, fixed_lead0, empty_fixed_lead),
  ]
  results = []

  print("\nBattery/Throughput Frequency Sweep")
  print("Energy model: E_total = total_cycles * 12.26 pJ")

  for f_hz in SWEEP_FREQUENCIES_HZ:
    cycles_per_sample = max(1, f_hz // SAMPLE_RATE)
    fifo_size = 2 * cycles_per_sample

    print(f"\n--- Frequency: {f_hz/1000:.0f} kHz | cycles/sample={cycles_per_sample} | fifo={fifo_size} ---")

    for mode_name, is_fixed, lead0_samples, lead1_samples in modes:
      _, _, thr0, thr1, bpm, clock, units0, units1 = run_simulation(
        lead0_samples,
        lead1_samples,
        is_fixed=is_fixed,
        clock_frequency_hz=f_hz,
        fifo_size=fifo_size,
        verbose=True,
      )

      uploader0 = units0[0]
      processed_lead0 = uploader0.sample_index
      total_cycles = clock.cycle

      busy, idle, stalled, util = aggregate_cycle_counters(units0)

      throughput_lead0_sps = (processed_lead0 * f_hz) / total_cycles if total_cycles > 0 else 0.0

      energy_total_pj = total_cycles * ENERGY_PER_CYCLE_PJ
      energy_total_uj = energy_total_pj / 1_000_000.0
      energy_per_sample_nj = (energy_total_pj / processed_lead0) / 1000.0 if processed_lead0 > 0 else 0.0

      results.append({
        "row_label": f"{mode_name} @ {f_hz/1000:.0f} kHz",
        "freq_khz": f_hz / 1000.0,
        "cycles_per_sample": cycles_per_sample,
        "fifo_size": fifo_size,
        "total_cycles": total_cycles,
        "busy_cycles": busy,
        "idle_cycles": idle,
        "stalled_cycles": stalled,
        "util": util,
        "throughput_lead0_sps": throughput_lead0_sps,
        "energy_total_uj": energy_total_uj,
        "energy_per_sample_nj": energy_per_sample_nj,
        "peaks0": len(thr0.peaks),
        "peaks1": len(thr1.peaks),
        "bpm": bpm,
      })

  print("\nSweep Summary")
  print(
    f"{'Mode @ Freq':<18} {'Cyc/S':>8} {'FIFO':>8} {'TotalCyc':>12} {'Busy':>12} {'Idle':>12} "
    f"{'Util%':>8} {'Thr(Lead0)':>11} {'E_total(uJ)':>12} {'E/samp(nJ)':>12}"
  )
  print("-" * 116)

  for r in results:
    print(
      f"{r['row_label']:<18} {r['cycles_per_sample']:>8,d} {r['fifo_size']:>8,d} {r['total_cycles']:>12,d} "
      f"{r['busy_cycles']:>12,d} {r['idle_cycles']:>12,d} {100*r['util']:>7.2f}% "
      f"{r['throughput_lead0_sps']:>11.3f} "
      f"{r['energy_total_uj']:>12.3f} {r['energy_per_sample_nj']:>12.3f}"
    )

  def print_mode_table(mode_name: str):
    mode_results = [r for r in results if r["row_label"].startswith(mode_name)]
    mode_results.sort(key=lambda r: r["cycles_per_sample"])

    print(f"\n{mode_name} Only Summary")
    print(
      f"{'Freq':<10} {'Cyc/S':>8} {'FIFO':>8} {'TotalCyc':>12} {'Busy':>12} {'Idle':>12} "
      f"{'Util%':>8} {'Thr(Lead0)':>11} {'E_total(uJ)':>12} {'E/samp(nJ)':>12}"
    )
    print("-" * 116)

    for r in mode_results:
      freq_label = f"{int(r['freq_khz'])} kHz"
      print(
        f"{freq_label:<10} {r['cycles_per_sample']:>8,d} {r['fifo_size']:>8,d} {r['total_cycles']:>12,d} "
        f"{r['busy_cycles']:>12,d} {r['idle_cycles']:>12,d} {100*r['util']:>7.2f}% "
        f"{r['throughput_lead0_sps']:>11.3f} {r['energy_total_uj']:>12.3f} {r['energy_per_sample_nj']:>12.3f}"
      )

  print_mode_table("Float")
  print_mode_table("Fixed")

  return results


if __name__ == "__main__":
  patient_number = 116
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_lead0, float_lead1 = load_data(record_path)
  fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)
  fixed_lead1 = (float_lead1 * FIXED_POINT_SCALE).astype(np.int32)

  actual_bpm = get_patient_bpm(patient_number)
  print(f"Patient #{patient_number}")
  print(f"Actual BPM (Database): {actual_bpm}")

  run_frequency_sweep(float_lead0, float_lead1, fixed_lead0, fixed_lead1)
