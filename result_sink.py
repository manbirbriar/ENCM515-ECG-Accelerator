from hardware_unit import HardwareUnit

class ResultSink(HardwareUnit):
    def __init__(self, name: str):
        super().__init__(name, latency_cycles=1)
        self.results = []

    def compute(self, data):
        self.results.append(data)
        return data