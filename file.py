import json
import os
from typing import Dict
from datetime import datetime, timezone

from singleton import Singleton


class FileHandler(metaclass=Singleton):
    def __init__(self):
        self.storage_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'storage')
        self.stats_cache: Dict[str, Dict[str, any]] = dict()
        self.day_key: str = datetime.utcfromtimestamp(
            datetime.timestamp(datetime.now().replace(tzinfo=timezone.utc).astimezone())).strftime(
            '%Y-%m-%d')
        os.makedirs("%s/stats" % self.storage_path, exist_ok=True)

    def read_statistics(self):
        try:
            with open("%s/stats/%s.json" % (self.storage_path, self.day_key), "r") as stats_file:
                file_data = stats_file.read()
                if file_data:
                    self.stats_cache = json.loads(file_data)
        except FileNotFoundError:
            with open("%s/stats/%s.json" % (self.storage_path, self.day_key), 'w+'):
                pass

    def append_statistics(self, data: Dict[str, any]):
        time_key: str = datetime.utcfromtimestamp(
            datetime.timestamp(datetime.now().replace(tzinfo=timezone.utc).astimezone())).strftime(
            '%Y-%m-%d %H:%M:%S')

        for name, stat in data.items():
            if name not in self.stats_cache.keys():
                self.stats_cache[name] = dict()
            self.stats_cache[name][time_key] = stat

    def write_statistics(self):
        with open("%s/stats/%s.json" % (self.storage_path, self.day_key), "w") as stats_file:
            stats_file.write(json.dumps(self.stats_cache))
