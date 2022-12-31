from typing import Any, Dict


class Singleton(type):
    _instances: Dict = {}

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
