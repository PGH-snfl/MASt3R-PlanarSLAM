"""Stage wall times; optional CUDA fences for diagnostic attribution."""
from contextlib import contextmanager
import time
import torch


class StageTimer:
    def __init__(self, synchronize=False):
        self.synchronize = synchronize
        self.times = {}

    def fence(self):
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name):
        self.fence()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.fence()
            self.times[name] = self.times.get(name, 0.) + 1000*(time.perf_counter()-start)


class DisplayGate:
    """Coalesce display updates without queueing old frames."""
    def __init__(self, hz):
        if hz <= 0:
            raise ValueError('Display frequency must be positive')
        self.period, self.next_update, self.dirty = 1./hz, 0., False

    def ready(self, now, changed=False):
        self.dirty |= bool(changed)
        if now < self.next_update:
            return False, False
        self.next_update = now + self.period
        publish, self.dirty = self.dirty, False
        return True, publish
