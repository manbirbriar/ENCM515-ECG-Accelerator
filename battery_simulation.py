import numpy as np

from config import BATTERY_CAPACITY_MAH, FIFO_SIZE, FIXED_POINT_SCALE, SAMPLE_RATE
from simulator import get_patient_bpm, load_data, run_simulation


ENERGY_PER_CYCLE_PJ = 12.26
BATTERY_VOLTAGE_V = 1.1
SWEEP_FREQUENCIES_HZ = [10_000, 25_000, 50_000, 100_000, 200_000]


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
  fixed_fifo_size = FIFO_SIZE

  for f_hz in SWEEP_FREQUENCIES_HZ:
    cycles_per_sample = max(1, f_hz // SAMPLE_RATE)
    fifo_size = fixed_fifo_size

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
      fifo0 = units0[1]
      processed_lead0 = uploader0.sample_index - fifo0.dropped_samples
      dropped_samples = fifo0.dropped_samples
      total_cycles = clock.cycle

      busy, idle, stalled, util = aggregate_cycle_counters(units0)

      throughput_lead0_sps = (processed_lead0 * f_hz) / total_cycles if total_cycles > 0 else 0.0

      energy_total_pj = total_cycles * ENERGY_PER_CYCLE_PJ
      energy_total_j = energy_total_pj * 1e-12
      runtime_s = total_cycles / f_hz if f_hz > 0 else 0.0
      avg_power_w = energy_total_j / runtime_s if runtime_s > 0 else 0.0
      battery_energy_j = (BATTERY_CAPACITY_MAH / 1000.0) * BATTERY_VOLTAGE_V * 3600.0
      battery_life_h = (battery_energy_j / avg_power_w) / 3600.0 if avg_power_w > 0 else 0.0
      battery_life_kh = battery_life_h / 1000.0

      results.append({
        "row_label": f"{mode_name} @ {f_hz/1000:.0f} kHz",
        "mode": mode_name,
        "freq_khz": f_hz / 1000.0,
        "throughput_lead0_sps": throughput_lead0_sps,
        "battery_life_kh": battery_life_kh,
        "dropped_samples": dropped_samples,
      })

  float_rows = [r for r in results if r["mode"] == "Float"]
  fixed_rows = [r for r in results if r["mode"] == "Fixed"]
  float_rows.sort(key=lambda r: r["freq_khz"])
  fixed_rows.sort(key=lambda r: r["freq_khz"])

  print("\nFloat Summary")
  print(f"{'Frequency':<10} {'BatteryLife(kh)':>15} {'Throughput(sps)':>16} {'Dropped':>10} {'Validity':>10}")
  print("-" * 70)
  for r in float_rows:
    validity = "Invalid" if r["dropped_samples"] > 0 else "Valid"
    print(
      f"{int(r['freq_khz']):<10} {r['battery_life_kh']:>15.3f} {r['throughput_lead0_sps']:>16.3f} "
      f"{r['dropped_samples']:>10,d} {validity:>10}"
    )

  print("\nFixed Summary")
  print(f"{'Frequency':<10} {'BatteryLife(kh)':>15} {'Throughput(sps)':>16} {'Dropped':>10} {'Validity':>10}")
  print("-" * 70)
  for r in fixed_rows:
    validity = "Invalid" if r["dropped_samples"] > 0 else "Valid"
    print(
      f"{int(r['freq_khz']):<10} {r['battery_life_kh']:>15.3f} {r['throughput_lead0_sps']:>16.3f} "
      f"{r['dropped_samples']:>10,d} {validity:>10}"
    )

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
