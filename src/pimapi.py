"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import requests
import yaml
import os


def load_conf(env):
    if env not in {'dev', 'int', 'rec', 'qat', 'prd'}:
        raise ValueError(f'Specified env : {env} not expected')
    conf_path = os.path.join(os.path.dirname(__file__), '../cfg/config.yaml')
    with open(conf_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    conf = data_loaded['cross-env']
    conf.update(data_loaded[env])
    return(conf)


def get_json_info(env, uid):
    requests.get()


class Config:

    def __init__(self):
        self.path = os.getcwd()
        stream = open(self.path+"/conf/config.yaml", 'r')
        data = yaml.load(stream)
        config_keys = data.keys()
        for k in config_keys:
            setattr(self, k, data.get(k))
        if (os.path.isfile(self.path+"/conf/config-override.yaml")):
            stream = open(self.path+"/conf/config-override.yaml", 'r')
            data = yaml.load(stream)
            config_keys = data.keys()
            for k in config_keys:
                setattr(self, k, data.get(k))


"""
config = Config()
And then any file that wants to use it:

from lib.project.Config import config
"""
