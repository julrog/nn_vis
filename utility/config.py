from utility.file import DictFile


class BaseConfig(dict):
    def __init__(self, name: str):
        super().__init__()
        self.dictFile: DictFile = DictFile(name, 'configs')
        self.dictFile.read_data(self)

    def set_defaults(self):
        pass

    def store(self):
        self.dictFile.write_data(self)
