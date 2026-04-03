# Fixed-point scaling
FIXED_POINT_SCALE = 2**15 - 1
FIXED_POINT_BITS = 15

# MIT-BIH dataset sample rate
SAMPLE_RATE: int = 360 # Hz

# Hardware clock frequency
# Real DSP hardware runs much faster than the sample rate
# This determines how many clock cycles are available per incoming sample
CLOCK_FREQUENCY: int = 36000 # Hz
CYCLES_PER_SAMPLE: int = CLOCK_FREQUENCY // SAMPLE_RATE # 1000 cycles per sample

# QRS complex typically lasts between 80ms - 120ms
# Use 150ms to ensure the entire complex is captured
# MWI_WINDOW_SIZE = 0.15s * 360 samples/s
MWI_WINDOW_SIZE: int = int(SAMPLE_RATE * 0.15) # 54 samples

# FIFO buffer between DataUploader and MACUnit
# Must be deep enough to absorb samples while MAC is busy
# Safe sizing: 2x cycles per sample
FIFO_SIZE: int = CYCLES_PER_SAMPLE * 2 # 200 slots

# Threshold unit peak detection parameters
MIN_PEAK_WIDTH: int = 3 # consecutive samples above threshold to confirm peak
REFRACTORY_SAMPLES: int = int(SAMPLE_RATE * 0.2) # 72 samples (200ms at 360Hz)

# Data recorder capacity
DATA_RECORDER_CAPACITY: int = 2500

# Number of samples to process. Without this, the simulation takes forever.
MAX_SAMPLES: int = 2500

# Operation cycle-cost table used by latency models.
# Fixed-point assumes integer datapath with 1-cycle arithmetic.
# Floating-point assumes a slower embedded FP datapath.
FIXED_ADD_CYCLES: int = 1
FIXED_SUB_CYCLES: int = 1
FIXED_MUL_CYCLES: int = 1
FIXED_SHIFT_CYCLES: int = 1
FIXED_COMPARE_CYCLES: int = 1

FLOAT_ADD_CYCLES: int = 31
FLOAT_SUB_CYCLES: int = 39
FLOAT_MUL_CYCLES: int = 26
FLOAT_DIV_CYCLES: int = 53
FLOAT_COMPARE_CYCLES: int = 13