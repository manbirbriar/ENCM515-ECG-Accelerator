from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from accelerator_sim import CORTEX_M4_PROFILE, CORTEX_M4F_PROFILE, compare_recorders, load_data, run_simulation
from circular_buffer import CircularDelayLine, CircularFIFO
from config import FIXED_POINT_SCALE, MWI_WINDOW_SIZE, REFRACTORY_PERIOD_SAMPLES, SAMPLE_RATE_HZ, get_cycle_table
from filter_mac_engine import FilterMacEngine
from hardware_unit import SampleToken
from mwi_unit import MWIUnit
from squaring_unit import SquaringUnit
from threshold_unit import PeakDetectorUnit


def reference_filter_mac(samples: list[float] | list[int], is_fixed: bool) -> list[float] | list[int]:
  low_pass_x_history = deque([0] * 12, maxlen=12)
  low_pass_y_1 = 0
  low_pass_y_2 = 0
  high_pass_x_history = deque([0] * 32, maxlen=32)
  high_pass_y_1 = 0
  derivative_x_history = deque([0] * 4, maxlen=4)
  outputs = []

  for sample in samples:
    x_n = int(sample) if is_fixed else float(sample)

    x_n6 = low_pass_x_history[6]
    x_n12 = low_pass_x_history[0]
    if is_fixed:
      low_pass_value = (int(low_pass_y_1) << 1) - int(low_pass_y_2) + (x_n - (int(x_n6) << 1) + int(x_n12))
    else:
      low_pass_value = (2.0 * float(low_pass_y_1)) - float(low_pass_y_2) + (x_n - (2.0 * float(x_n6)) + float(x_n12))
    low_pass_y_2, low_pass_y_1 = low_pass_y_1, low_pass_value
    low_pass_x_history.append(x_n)

    hp_n16 = high_pass_x_history[16]
    hp_n17 = high_pass_x_history[15]
    hp_n32 = high_pass_x_history[0]
    if is_fixed:
      high_pass_value = int(high_pass_y_1) + (-(int(low_pass_value) >> 5) + int(hp_n16) - int(hp_n17) + (int(hp_n32) >> 5))
    else:
      high_pass_value = float(high_pass_y_1) + (-(float(low_pass_value) / 32.0) + float(hp_n16) - float(hp_n17) + (float(hp_n32) / 32.0))
    high_pass_y_1 = high_pass_value
    high_pass_x_history.append(low_pass_value)

    der_n1 = derivative_x_history[3]
    der_n3 = derivative_x_history[1]
    der_n4 = derivative_x_history[0]
    if is_fixed:
      derivative_value = ((2 * int(high_pass_value)) + int(der_n1) - int(der_n3) - (2 * int(der_n4))) >> 3
    else:
      derivative_value = 0.125 * ((2.0 * float(high_pass_value)) + float(der_n1) - float(der_n3) - (2.0 * float(der_n4)))
    derivative_x_history.append(high_pass_value)
    outputs.append(derivative_value)

  return outputs


def reference_squaring(samples: list[float] | list[int], is_fixed: bool) -> list[float] | list[int]:
  if is_fixed:
    return [((int(sample) * int(sample)) >> 15) for sample in samples]
  return [float(sample) * float(sample) for sample in samples]


def reference_mwi(samples: list[float] | list[int], is_fixed: bool) -> list[float] | list[int]:
  window = deque([0] * MWI_WINDOW_SIZE, maxlen=MWI_WINDOW_SIZE)
  running_sum = 0
  outputs = []
  for sample in samples:
    current = int(sample) if is_fixed else float(sample)
    running_sum += current - window[0]
    window.append(current)
    outputs.append((running_sum // MWI_WINDOW_SIZE) if is_fixed else (running_sum / MWI_WINDOW_SIZE))
  return outputs


def reference_peak_detector(samples: list[float] | list[int], is_fixed: bool) -> tuple[list[float] | list[int], list[int]]:
  recent_values = deque([0 if is_fixed else 0.0, 0 if is_fixed else 0.0], maxlen=2)
  if is_fixed:
    spki = 2 * FIXED_POINT_SCALE
    npki = FIXED_POINT_SCALE // 2
    threshold = FIXED_POINT_SCALE
  else:
    spki = 2.0
    npki = 0.5
    threshold = 1.0
  last_peak_sample = -REFRACTORY_PERIOD_SAMPLES
  pulses = []
  peaks = []

  for sample_id, sample in enumerate(samples):
    current = int(sample) if is_fixed else float(sample)
    previous = recent_values[-1]
    two_back = recent_values[0]
    pulse = 0 if is_fixed else 0.0
    is_local_peak = previous > two_back and previous >= current
    candidate_sample = sample_id - 1
    if is_local_peak and candidate_sample >= 0:
      outside_refractory = (candidate_sample - last_peak_sample) > REFRACTORY_PERIOD_SAMPLES
      if previous > threshold and outside_refractory:
        peaks.append(candidate_sample)
        last_peak_sample = candidate_sample
        if is_fixed:
          spki = (int(previous) + (7 * int(spki))) >> 3
        else:
          spki = (0.125 * previous) + (0.875 * spki)
        pulse = 1 if is_fixed else 1.0
      else:
        if is_fixed:
          npki = (int(previous) + (7 * int(npki))) >> 3
        else:
          npki = (0.125 * previous) + (0.875 * npki)
    else:
      if is_fixed:
        npki = (int(current) + (7 * int(npki))) >> 3
      else:
        npki = (0.125 * current) + (0.875 * npki)

    if is_fixed:
      threshold = int(npki) + ((int(spki) - int(npki)) >> 2)
    else:
      threshold = npki + (0.25 * (spki - npki))
    recent_values.append(current)
    pulses.append(pulse)

  return pulses, peaks


class StreamingAcceleratorTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.float_samples = load_data("ecg_data/patient_116/116")[:4000]

  def _run_stage(self, unit, samples) -> list[float] | list[int]:
    outputs = []
    for sample_id, sample in enumerate(samples):
      token = SampleToken(sample_id=sample_id, value=sample, ingress_cycle=sample_id + 1)
      outputs.append(unit.process_token(token, sample_id + 1).value)
    return outputs

  def test_filter_mac_matches_reference_float(self) -> None:
    samples = self.float_samples[:256].tolist()
    unit = FilterMacEngine("filter_mac_engine", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), False)
    self.assertTrue(np.allclose(self._run_stage(unit, samples), reference_filter_mac(samples, False)))

  def test_peak_detector_matches_reference(self) -> None:
    samples = reference_mwi(reference_squaring(reference_filter_mac(self.float_samples[:512].tolist(), False), False), False)
    expected_pulses, expected_peaks = reference_peak_detector(samples, False)
    detector = PeakDetectorUnit("peak_detector", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), SAMPLE_RATE_HZ, REFRACTORY_PERIOD_SAMPLES, False)
    actual_pulses = self._run_stage(detector, samples)
    self.assertEqual(actual_pulses, expected_pulses)
    self.assertEqual(detector.peaks, expected_peaks)

  def test_circular_buffer_wraparound(self) -> None:
    delay_line = CircularDelayLine[int](capacity=4, zero_value=0)
    for value in [10, 20, 30, 40, 50]:
      delay_line.append(value)

    fifo = CircularFIFO[int](capacity=3)
    self.assertTrue(fifo.push(1))
    self.assertTrue(fifo.push(2))
    self.assertTrue(fifo.push(3))
    self.assertEqual(fifo.pop(), 1)
    self.assertTrue(fifo.push(4))

    self.assertEqual(delay_line.delay(1), 50)
    self.assertEqual(delay_line.delay(2), 40)
    self.assertEqual(delay_line.delay(4), 20)
    self.assertEqual(fifo.pop(), 2)
    self.assertEqual(fifo.pop(), 3)
    self.assertEqual(fifo.pop(), 4)

  def test_fixed_pipeline_uses_integer_values(self) -> None:
    fixed_input = (self.float_samples[:64] * FIXED_POINT_SCALE).astype(np.int32)
    filter_unit = FilterMacEngine("filter_mac_engine", get_cycle_table(CORTEX_M4_PROFILE, is_fixed_point=True), True)
    square_unit = SquaringUnit("squaring_unit", get_cycle_table(CORTEX_M4_PROFILE, is_fixed_point=True), True)
    mwi_unit = MWIUnit("moving_window_integrator", get_cycle_table(CORTEX_M4_PROFILE, is_fixed_point=True), MWI_WINDOW_SIZE, True)
    peak_unit = PeakDetectorUnit("peak_detector", get_cycle_table(CORTEX_M4_PROFILE, is_fixed_point=True), SAMPLE_RATE_HZ, REFRACTORY_PERIOD_SAMPLES, True)

    token = SampleToken(sample_id=0, value=int(fixed_input[0]), ingress_cycle=1)
    filter_out = filter_unit.process_token(token, 1)
    square_out = square_unit.process_token(filter_out, 1)
    mwi_out = mwi_unit.process_token(square_out, 1)
    peak_out = peak_unit.process_token(mwi_out, 1)

    self.assertIsInstance(filter_out.value, int)
    self.assertIsInstance(filter_out.metadata["low_pass"], int)
    self.assertIsInstance(filter_out.metadata["high_pass"], int)
    self.assertIsInstance(filter_out.metadata["derivative"], int)
    self.assertIsInstance(square_out.value, int)
    self.assertIsInstance(mwi_out.value, int)
    self.assertIsInstance(peak_out.value, int)
    self.assertIsInstance(peak_unit.spki, int)
    self.assertIsInstance(peak_unit.npki, int)
    self.assertIsInstance(peak_unit.threshold, int)

  def test_float_pipeline_uses_float_values(self) -> None:
    samples = self.float_samples[:64].astype(np.float32)
    filter_unit = FilterMacEngine("filter_mac_engine", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), False)
    square_unit = SquaringUnit("squaring_unit", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), False)
    mwi_unit = MWIUnit("moving_window_integrator", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), MWI_WINDOW_SIZE, False)
    peak_unit = PeakDetectorUnit("peak_detector", get_cycle_table(CORTEX_M4F_PROFILE, is_fixed_point=False), SAMPLE_RATE_HZ, REFRACTORY_PERIOD_SAMPLES, False)

    token = SampleToken(sample_id=0, value=float(samples[0]), ingress_cycle=1)
    filter_out = filter_unit.process_token(token, 1)
    square_out = square_unit.process_token(filter_out, 1)
    mwi_out = mwi_unit.process_token(square_out, 1)
    peak_out = peak_unit.process_token(mwi_out, 1)

    self.assertIsInstance(filter_out.value, float)
    self.assertIsInstance(filter_out.metadata["low_pass"], float)
    self.assertIsInstance(filter_out.metadata["high_pass"], float)
    self.assertIsInstance(filter_out.metadata["derivative"], float)
    self.assertIsInstance(square_out.value, float)
    self.assertIsInstance(mwi_out.value, float)
    self.assertIsInstance(peak_out.value, float)
    self.assertIsInstance(peak_unit.spki, float)
    self.assertIsInstance(peak_unit.npki, float)
    self.assertIsInstance(peak_unit.threshold, float)

  def test_cortex_m4_fixed_processes_without_drop(self) -> None:
    results = run_simulation(self.float_samples[:2000], is_fixed=True, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=1_000_000)
    metrics = results["metrics"]
    self.assertEqual(metrics["accepted_samples"], metrics["attempted_samples"])
    self.assertEqual(metrics["dropped_samples"], 0)
    self.assertEqual(metrics["input_buffer_overflows"], 0)
    self.assertGreater(metrics["sustainable_throughput_margin_vs_input"], 1.0)

  def test_cortex_m4f_fast_clock_creates_large_idle_budget(self) -> None:
    results = run_simulation(self.float_samples[:2000], is_fixed=False, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=80_000_000)
    metrics = results["metrics"]
    self.assertGreater(metrics["front_end_idle_cycles"], metrics["accepted_samples"])
    self.assertEqual(metrics["dropped_samples"], 0)

  def test_cortex_m4_float_is_slower_than_fixed(self) -> None:
    float_results = run_simulation(self.float_samples[:512], is_fixed=False, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=1_000_000)
    fixed_results = run_simulation(self.float_samples[:512], is_fixed=True, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=1_000_000)
    self.assertGreater(
      float_results["metrics"]["average_end_to_end_latency_cycles"],
      fixed_results["metrics"]["average_end_to_end_latency_cycles"],
    )

  def test_cortex_m4f_float_is_faster_than_m4_soft_float(self) -> None:
    m4_float_results = run_simulation(self.float_samples[:512], is_fixed=False, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=1_000_000)
    m4f_float_results = run_simulation(self.float_samples[:512], is_fixed=False, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=1_000_000)
    self.assertGreater(
      m4_float_results["metrics"]["average_end_to_end_latency_cycles"],
      m4f_float_results["metrics"]["average_end_to_end_latency_cycles"],
    )

  def test_m4f_float_and_fixed_remain_aligned(self) -> None:
    float_results = run_simulation(self.float_samples[:2000], is_fixed=False, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=1_000_000)
    fixed_results = run_simulation(self.float_samples[:2000], is_fixed=True, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=1_000_000)
    rmse_by_stage = compare_recorders(float_results["recorders"], fixed_results["recorders"])
    self.assertLess(abs(float_results["peak_detector_unit"].get_bpm() - fixed_results["peak_detector_unit"].get_bpm()), 1.0)
    self.assertLess(rmse_by_stage["peak_detector"], 0.1)

  def test_m4_soft_float_and_m4f_float_remain_aligned(self) -> None:
    m4_soft_float_results = run_simulation(self.float_samples[:2000], is_fixed=False, core_profile=CORTEX_M4_PROFILE, accel_clock_hz=1_000_000)
    m4f_float_results = run_simulation(self.float_samples[:2000], is_fixed=False, core_profile=CORTEX_M4F_PROFILE, accel_clock_hz=1_000_000)
    rmse_by_stage = compare_recorders(
      m4f_float_results["recorders"],
      m4_soft_float_results["recorders"],
      candidate_is_fixed=False,
    )
    self.assertLess(rmse_by_stage["filter_mac"], 1e-6)
    self.assertLess(rmse_by_stage["mwi"], 1e-6)
    self.assertLess(rmse_by_stage["peak_detector"], 1e-6)


if __name__ == "__main__":
  unittest.main()
