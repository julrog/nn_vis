from enum import Enum, IntFlag, auto


class ImportanceType(IntFlag):
    CENTERING = auto()
    GAMMA = auto()
    L1 = auto()
    L2 = auto()


def get_importance_type_name(importance_type: ImportanceType) -> str:
    name: str = ''
    name = name + ('beta_' if importance_type &
                   ImportanceType.CENTERING else 'nobeta_')
    name = name + ('gammaone' if importance_type &
                   ImportanceType.GAMMA else 'gammazero')
    if importance_type & ImportanceType.L1:
        name = name + '_' + 'l1'
    if importance_type & ImportanceType.L1 and importance_type & ImportanceType.L2:
        name = name + 'l2'
    elif importance_type & ImportanceType.L2:
        name = name + '_' + 'l2'
    return name


class ImportanceCalculation(Enum):
    BNN_EDGE = 1
    BNN_ONLY = 2
    EDGE_ONLY = 3
