import time

class Benchmark:
    def __init__(self, description="Done"):
        self.description = description
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        print(f'{self.description}: {self.end_time - self.start_time:.4f} sec')