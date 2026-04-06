# Fixed-point scaling
FIXED_POINT_BITS = 15
FIXED_POINT_SCALE = 2**FIXED_POINT_BITS - 1

# MIT-BIH dataset sample rate
SAMPLE_RATE: int = 360 # Hz

# QRS complex typically lasts between 80ms - 120ms
# 150ms to ensure the entire complex is captured
# MWI_WINDOW_SIZE = 0.15s * 360 samples/s
MWI_WINDOW_SIZE: int = int(SAMPLE_RATE * 0.15) # 54 samples

# Threshold unit peak detection parameters
MIN_PEAK_WIDTH: int = 3 # Consecutive samples above threshold to confirm peak
REFRACTORY_SAMPLES: int = int(SAMPLE_RATE * 0.2) # 72 samples (200ms at 360Hz)

# Data recorder capacity
DATA_RECORDER_CAPACITY: int = 2500

# Number of samples to process
MAX_SAMPLES: int = 2500

# Frequency Testing
SWEEP_FREQUENCIES_HZ = [3600, 36000, 360000]

# MAC Unit SIMD Vectorization
VECTOR_LENGTHS = [1, 4, 8, 12]

# Fixed-point latencies model a ARM Cortex-M4
# https://developer.arm.com/documentation/ddi0439/b/CHDDIGAC
FIXED_ADD_CYCLES: int = 1
FIXED_SUB_CYCLES: int = 1
FIXED_MUL_CYCLES: int = 1
FIXED_DIV_CYCLES: int = 7
FIXED_SHIFT_CYCLES: int = 1
FIXED_COMPARE_CYCLES: int = 1
FIXED_MAC_CYCLES: int = 1

# Floating-point latencies model a software FP library (no hardware FPU)
# https://blog.segger.com/wp-content/uploads/2019/11/Bench_Comparison.pdf
FLOAT_ADD_CYCLES: int = 52
FLOAT_SUB_CYCLES: int = 56
FLOAT_MUL_CYCLES: int = 154
FLOAT_DIV_CYCLES: int = 144
FLOAT_COMPARE_CYCLES: int = 51
FLOAT_MAC_CYCLES: int = 206

# https://developer.arm.com/Processors/Cortex-M4
# Battery and power model parameters
# We are emulating a ARM Cortex-M4 180ULL at 1.8V which is a typical wearable configuration
BATTERY_CAPACITY_MAH: int = 225  # CR2032 coin cell
BATTERY_VOLTAGE: float = 1.8  # Volts (ARM 180ULL)
DYNAMIC_POWER_UW_PER_MHZ: float = 151  # µW/MHz at 1.8V (ARM 180ULL)

# NOTE: These values are no longer used from here

# Real DSP hardware runs much faster than the sample rate
# A multiple of SAMPLE_RATE which ensures that CYCLES_PER_SAMPLE has no rounding error
CLOCK_FREQUENCY: int = 3600 # Hz 

# How many clock cycles are available per incoming sample
CYCLES_PER_SAMPLE: int = CLOCK_FREQUENCY // SAMPLE_RATE

# FIFO buffer between DataUploader and MACUnit
FIFO_SIZE: int = CYCLES_PER_SAMPLE * 2