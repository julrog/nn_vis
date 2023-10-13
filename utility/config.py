from typing import Optional

from utility.file import DictFile


class BaseConfig(dict):
    def __init__(self, config_type: str, name: Optional[str] = None) -> None:
        super().__init__()
        self.dictFile: DictFile = DictFile(config_type, 'configs') if name is None else DictFile(
            name + '_' + config_type, 'configs')

        self.dictFile.read_data(self)

    def set_defaults(self) -> None:
        pass

    def store(self) -> None:
        self.dictFile.write_data(self)
