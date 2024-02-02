import os

from omegaconf import DictConfig
from omegaconf import OmegaConf


class ConfigSingleton:
    _config = None

    @staticmethod
    def load_config():
        override_config_path = "/home/ekin/Scenic/examples/ekin/experiment/default_config.yaml"
        if os.path.exists(override_config_path):
            override_cfg = OmegaConf.load(override_config_path)

        #config.carla.host = os.environ.get("CARLA_CONTAINER", config.carla.host)
        ConfigSingleton._config = override_cfg

    @staticmethod
    def get_config():
        if ConfigSingleton._config is None:
            raise ValueError("a has not been loaded")
        return ConfigSingleton._config

'''
class ConfigProxy:
    @staticmethod
    def __getattr__(attr):
        config = ConfigSingleton.load_config()
        return getattr(config, attr)

    def to_dict(self):
        config = ConfigSingleton.get_config()
        return OmegaConf.to_container(config)

    def to_dictconfig(self):
        config = ConfigSingleton.get_config()
        return OmegaConf.create(config)
'''

cfg = OmegaConf.load("/home/ekin/Scenic/examples/ekin/experiment/default_config.yaml")
