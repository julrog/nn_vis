import math


class AdvectionProgress:
    def __init__(self, bandwidth: float, bandwidth_reduction: float, limit: float):
        self.iteration: int = 0
        self.bandwidth: float = bandwidth
        self.bandwidth_reduction: float = bandwidth_reduction
        self.limit_reached: bool = False
        self.limit: float = limit
        self.current_bandwidth: float = self.bandwidth
        self.advection_direction: float = 1.0
        self.importance_similarity: float = 0.0

    def reset(self):
        self.iteration = 0
        self.limit_reached = False

    def iterate(self):
        self.iteration += 1
        self.current_bandwidth = self.bandwidth * math.pow(self.bandwidth_reduction, self.iteration)

        self.importance_similarity = 1.0 - (self.current_bandwidth - self.limit) / (self.bandwidth - self.limit)
        self.importance_similarity = math.sqrt(max(self.importance_similarity * 0.9, 0.00))

        if self.current_bandwidth < self.limit:
            self.limit_reached = True

    def get_advection_strength(self) -> float:
        return self.current_bandwidth * self.advection_direction

    def get_bandwidth_reduction(self):
        return math.pow(self.bandwidth_reduction, self.iteration)
