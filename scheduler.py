from hardware_unit import HardwareUnit
from sample_queue import SampleQueue

# For each tick, if a full window is ready and at least one lane is free,
# the scheduler extracts the window from the buffer and sends it to the first available lane
class Scheduler(HardwareUnit):
  # def __init__(self, name: str, lane_queues: list[tuple[SampleQueue, HardwareUnit]]):
  def __init__(self, name: str, sample_queue: SampleQueue, lanes: list[HardwareUnit]):
    super().__init__(name, latency_cycles=1)

    # self.lane_queues = lane_queues
    self.sample_queue: SampleQueue = sample_queue
    self.lanes: list[HardwareUnit] = lanes

    self.stall_count: int = 0
    self.dispatched_window_count: int = 0
    self.stalled: bool = False

    self.current_sample_index = 0

  # If a window is full and a lane is free, dispatch window, else, stall
  def tick(self, current_cycle: int) -> None:
    self.current_cycle = current_cycle
    self.stalled = False

    if not self.sample_queue.window_ready():
      return

    free_lane = self.find_free_lane()

    if not free_lane:
      self.stalled = True
      self.stall_count += 1
      return

    window = self.sample_queue.get_window()

    if self.recorder:
      self.recorder.record(window)

    free_lane.input_data = window
    self.dispatched_window_count += 1

  # def tick(self, current_cycle: int) -> None:
  #   self.current_cycle = current_cycle
  #   self.stalled = False

  #   for queue, lane in self.lane_queues:
  #     if queue.window_ready() and lane.is_available():
  #       window = queue.get_window()
  #       if self.recorder:
  #         self.recorder.record(window)
  #       lane.input_data = window
  #       self.dispatched_window_count += 1
  #     elif queue.window_ready() and not lane.is_available():
  #       self.stalled = True
  #       self.stall_count += 1

  # Return the first available lane or None if all are busy
  def find_free_lane(self) -> HardwareUnit | None:
    for lane in self.lanes:
      if lane.is_available():
        return lane
    return None

  # Unused
  def compute(self, data: list) -> list:
    return data

  def is_stalled(self) -> bool:
    return self.stalled

  def __repr__(self) -> str:
    return f"<Scheduler name={self.name} dispatched={self.dispatched_window_count} stall_count={self.stall_count}>"