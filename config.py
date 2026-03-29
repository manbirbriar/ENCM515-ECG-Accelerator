# TODO: Confirm that this is valid
FIXED_POINT_SCALE = 2**15 - 1

# MIT-BIH dataset sample rate is 360Hz
SAMPLE_RATE: int = 360

# QRS complex typically lasts between 80ms - 120ms
# Use 150ms ensure the entire complex is captured
# MWI_WINDOW_SIZE = 0.15s * 360 samples/s
# MWI_WINDOW_SIZE: int = 0.15 * SAMPLE_RATE
MWI_WINDOW_SIZE: int = 54

WINDOW_SIZE: int = 72
# WINDOW_SIZE: int = SAMPLE_RATE * 0.2

HOP_SIZE: int = WINDOW_SIZE

# TODO: Not sure how large the queue_size should be
QUEUE_SIZE: int = 512

DATA_RECORDER_CAPACITY: int = 2500

VECTOR_WIDTH: int = 4