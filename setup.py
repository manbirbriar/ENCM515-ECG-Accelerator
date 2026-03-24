import wfdb
import numpy as np

FIXED_POINT_SCALE = 2**15 - 1

# TODO: Confirm that this is working correctly
def load_float_samples(record_path: str, channel: int = 0) -> np.ndarray:
  record = wfdb.rdrecord(record_path)
  raw = record.p_signal[:, channel]
  return raw.astype(np.float32)

# TODO: Confirm that this is working correctly
def load_fixed_samples(record_path: str, channel: int = 0) -> np.ndarray:
  record = wfdb.rdrecord(record_path)
  raw = record.p_signal[:, channel]
  normalised = raw / np.max(np.abs(raw))
  return (normalised * FIXED_POINT_SCALE).astype(np.int16)

# TODO: Confirm that this is working correctly
if __name__ == "__main__":
  patient_number = input("Enter the paitient number: ")

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_float_samples(record_path)
  fixed_samples = load_fixed_samples(record_path)

  print(f"Float samples: dtype: {float_samples.dtype}, shape: {float_samples.shape}, range: [{float_samples.min()}, {float_samples.max()}]")
  print(f"Fixed samples: dtype: {fixed_samples.dtype}, shape: {fixed_samples.shape}, range: [{fixed_samples.min()}, {fixed_samples.max()}]")