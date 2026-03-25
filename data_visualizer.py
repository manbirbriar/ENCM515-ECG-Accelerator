import matplotlib.pyplot as plt
import numpy as np

# stores and plots max_samples signals for each pipeline stage
class DataVisualizer:
  def __init__(self, max_samples: int = 5000):
    self.stages = {}
    self.MAX_SAMPLES = max_samples

  def add_snapshot(self, stage_name: str, data: list | np.ndarray):
    if stage_name not in self.stages:
      self.stages[stage_name] = []

    current_count = len(self.stages[stage_name])
    if current_count >= self.MAX_SAMPLES:
      return

    space_remaining = self.MAX_SAMPLES - current_count
    
    new_data = list(data)[:space_remaining]
    self.stages[stage_name].extend(new_data)

  def plot(self, title="ECG Pipeline Signals"):
    num_stages = len(self.stages)
    if num_stages == 0:
      return

    fig, axes = plt.subplots(num_stages, 1, figsize=(14, 4 * num_stages), sharex=True)
    
    if num_stages == 1:
      axes = [axes]

    for ax, (name, data) in zip(axes, self.stages.items()):
      ax.plot(data, label=name, color="blue", linewidth=1)
      ax.set_ylabel("Amplitude")
      ax.set_title(name)
      ax.legend(loc="upper right")

    plt.xlabel("Samples")
    plt.show()