from argparse import ArgumentParser
from typing import Self
from pathlib import Path
from utils import read_json


class DeviceConfig(object):

    def __init__(self, type="gpu", total=1):
        self._type = type
        self._total = total

    def __getattr__(self, item):
        if item == "type":
            return self._type
        if item == "total":
            return self._total
        return None

    def __str__(self):
        return str(self.__dict__)


class BaseConfig(object):

    def __init__(self, type, args=None):
        assert type is not None
        self._type = type
        self._args = args

    def __getitem__(self, item):
        if item == "type":
            return self._type
        if item == "args":
            return dict() if self._args is None else dict(self._args)
        return None

    def __str__(self):
        return str(self.__dict__)


class ConfigParser(object):

    def __init__(self, config):
        self._config = config
        self._device = DeviceConfig(**config["device"])
        self._model = BaseConfig(**config["model"])
        self._data_loader = BaseConfig(**config["data_loader"])

    @classmethod
    def from_args(cls, args: ArgumentParser) -> Self:
        if not isinstance(args, tuple):
            args = args.parse_args()
        config = read_json(args.config)
        return cls(config)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, item):
        if item == "device":
            return self._device
        if item == "model":
            return self._model
        if item == "data_loader":
            return self._data_loader
        return None
