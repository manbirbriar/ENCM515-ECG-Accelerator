# TODO: Confirm that this is valid
FIXED_POINT_SCALE = 2**15 - 1
FIXED_POINT_BITS = 15

# MIT-BIH dataset sample rate is 360Hz
SAMPLE_RATE: int = 360

# QRS complex typically lasts between 80ms - 120ms
# Use 150ms ensure the entire complex is captured
# MWI_WINDOW_SIZE = 0.15s * 360 samples/s
MWI_WINDOW_SIZE: int = int(SAMPLE_RATE * 0.15) # 54

WINDOW_SIZE: int = int(SAMPLE_RATE * 0.2) # 72

HOP_SIZE: int = WINDOW_SIZE

# TODO: Not sure how large the queue_size should be
QUEUE_SIZE: int = int(WINDOW_SIZE * 10) # 720

DATA_RECORDER_CAPACITY: int = 2500

VECTOR_WIDTH: int = 12

# Operation cycle-cost table used by latency models.
# Fixed-point assumes integer datapath with 1-cycle arithmetic.
# Floating-point assumes a slower embedded FP datapath.
FIXED_ADD_CYCLES: int = 1
FIXED_SUB_CYCLES: int = 1
FIXED_MUL_CYCLES: int = 1
FIXED_SHIFT_CYCLES: int = 1
FIXED_COMPARE_CYCLES: int = 1

FLOAT_ADD_CYCLES: int = 2
FLOAT_SUB_CYCLES: int = 2
FLOAT_MUL_CYCLES: int = 3
FLOAT_SHIFT_CYCLES: int = 1
FLOAT_COMPARE_CYCLES: int = 1