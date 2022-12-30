from enum import Enum
from typing import List


class StatisticLink(Enum):
    EDGE_COUNT = 0
    SAMPLE_COUNT = 1
    CELL_COUNT = 2
    PRUNED_EDGES = 3
    FPS = 4


STATISTIC_NAME: List[str] = ['Edges', 'Samples',
                             'Grid Cells', 'Pruned Edges', 'FPS']
