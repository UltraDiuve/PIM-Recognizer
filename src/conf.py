"""Configuration Module

This module provides a class that enables to fetch configuration stored in the
cfg folder and make it available to other python modules.
"""

import yaml
import os


class Config:

    def __init__(self, env):
        if env not in {'dev', 'int', 'rec', 'qat', 'prd'}:
            raise ValueError(f'Specified env : {env} not expected')
        self.env = env
        self.path = os.path.join(os.path.dirname(__file__),
                                 '../cfg/config.yaml')
        stream = open(self.path, 'r')
        data = yaml.safe_load(stream)
        config_keys = data['cross-env'].keys()
        for k in config_keys:
            setattr(self, k, data['cross-env'][k])
        config_keys = data[env].keys()
        for k in config_keys:
            setattr(self, k, data[env][k])


"""
from src import conf
cfg = conf.Config('rec')
"""
