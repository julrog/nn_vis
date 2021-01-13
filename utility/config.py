from utility.file import DictFile


class BaseConfig(dict):
    def __init__(self, config_type: str, name: str = None):
        super().__init__()
        if name is None:
            self.dictFile: DictFile = DictFile(config_type, 'configs')
        else:
            self.dictFile: DictFile = DictFile(name + '_' + config_type, 'configs')

        self.dictFile.read_data(self)

    def set_defaults(self):
        pass

    def store(self):
        self.dictFile.write_data(self)
