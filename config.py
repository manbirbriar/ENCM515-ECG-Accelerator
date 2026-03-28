# TODO: Confirm that this is valid
FIXED_POINT_SCALE = 2**15 - 1

SAMPLE_RATE = 360

# MIT-BIH dataset sample rate is 360Hz
# TODO: I've gone with a 0.2s wide window, but I cannot justify it
# window_size = 360 samples/s * 0.2s = 72 samples
WINDOW_SIZE: int = 72 # 360 * 0.2s

# TODO: Find out why 36 was giving bad results
HOP_SIZE: int = 72 # 50% overlap

# TODO: Not sure how large the queue_size should be
QUEUE_SIZE: int = 512

DATA_RECORDER_CAPACITY: int = 5000