from typing import Any, Callable, Dict, Type, Union

from definitions import CameraPose, ProcessRenderMode

NN_VIS_TYPES: Dict[str, Type[Union[ProcessRenderMode, CameraPose]]] = {
    'ProcessRenderMode': ProcessRenderMode,
    'CameraPose': CameraPose
}


def nnvis_to_str(value: Any) -> str:
    for nnvis_type_name, nnvis_type in NN_VIS_TYPES.items():
        if isinstance(value, nnvis_type):
            if value.name:
                return nnvis_type_name + '.' + value.name
            else:
                return str(value)
    return value


def convert_values(obj: Any, convert: Callable) -> Any:
    if isinstance(obj, list):
        return [convert_values(i, convert) for i in obj]
    if not isinstance(obj, dict):
        return convert(obj)
    return {k: convert_values(v, convert) for k, v in obj.items()}


def str_to_nnvis(value: Any) -> Union[ProcessRenderMode, CameraPose, Any]:
    if isinstance(value, str):
        for nnvis_type in NN_VIS_TYPES.keys():
            if nnvis_type in value:
                name, combined_members = value.split('.')
                if '|' in combined_members:
                    flag = 0
                    for member in combined_members.split('|'):
                        flag = flag | getattr(NN_VIS_TYPES[nnvis_type], member)
                    return flag
                else:
                    return getattr(NN_VIS_TYPES[nnvis_type], combined_members)
    return value
