"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import requests
import os
import conf


class requester(object):
    """Requester class to retrieve information from PIM
    """
    def __init__(self, env):
        self.cfg = conf.Config(env)

    def get_info_from_uid(self, uid):
        url = self.cfg.baseurl + self.cfg.suffixid
        print(url)
