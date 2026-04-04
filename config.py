# Fixed-point scaling
FIXED_POINT_BITS = 15
FIXED_POINT_SCALE = 2**FIXED_POINT_BITS - 1

# MIT-BIH dataset sample rate
SAMPLE_RATE: int = 360 # Hz

# Hardware clock frequency
# Real DSP hardware runs much faster than the sample rate
# A multiple of SAMPLE_RATE which ensures that CYCLES_PER_SAMPLE has no rounding error
CLOCK_FREQUENCY: int = 200160 # Hz 
# How many clock cycles are available per incoming sample
CYCLES_PER_SAMPLE: int = CLOCK_FREQUENCY // SAMPLE_RATE

# QRS complex typically lasts between 80ms - 120ms
# Use 150ms to ensure the entire complex is captured
# MWI_WINDOW_SIZE = 0.15s * 360 samples/s
MWI_WINDOW_SIZE: int = int(SAMPLE_RATE * 0.15) # 54 samples

# FIFO buffer between DataUploader and MACUnit
# Must be deep enough to absorb samples while MAC is busy
FIFO_SIZE: int = CYCLES_PER_SAMPLE * 2

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
FIXED_DIV_CYCLES: int = 7
FIXED_SHIFT_CYCLES: int = 1
FIXED_COMPARE_CYCLES: int = 1
FIXED_MAC_CYCLES: int = 1

FLOAT_ADD_CYCLES: int = 31
FLOAT_SUB_CYCLES: int = 39
FLOAT_MUL_CYCLES: int = 26
FLOAT_DIV_CYCLES: int = 53
FLOAT_COMPARE_CYCLES: int = 13
FLOAT_MAC_CYCLES: int = 60

# Battery and power model parameters
# ARM Cortex-M4 40LP @ 1.1V (typical wearable configuration)
BATTERY_CAPACITY_MAH: int = 225  # CR2032 coin cell
BATTERY_VOLTAGE: float = 1.1  # Volts (ARM 40LP nominal)
DYNAMIC_POWER_UW_PER_MHZ: float = 12.26  # µW/MHz @ 1.1V (from ARM datasheet)