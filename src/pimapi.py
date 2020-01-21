"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import requests
import os
import json

from . import conf


class Requester(object):
    """Requester class to retrieve information from PIM
    """
    def __init__(self, env, proxies=None):
        self.cfg = conf.Config(env)
        self.proxies = proxies

    def get_info_from_uid(self, uid, nx_properties='*'):
        """Requests an object data from PIM

        nx_properties describes which schemes are to be retrieved. Setting it
        to '*' means all data. Setting it to None returns only Nuxeo standard
        data.
        """
        headers = {'Content-Type': 'application/json',
                   'X-NXproperties': nx_properties}
        self.uid = uid
        url = self.cfg.baseurl + self.cfg.suffixid + uid
        self.result = requests.get(url,
                                   proxies=self.proxies,
                                   headers=headers,
                                   auth=(self.cfg.user, self.cfg.password))
        self.result.raise_for_status()

    def check_if_fetched(self):
        if self.result is None:
            raise RuntimeError('No data has been fetched yet')
        self.result.raise_for_status()

    def set_dump_path(self):
        self.check_if_fetched()
        self.path = os.path.join(os.path.dirname(__file__),
                                 '..',
                                 'dumps',
                                 self.cfg.env,
                                 self.uid)

    def dump_data(self, path=None):
        self.check_if_fetched()
        if path is None:
            self.set_dump_path()
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path, 'w+') as outfile:
            json.dump(self.result.json(), outfile)

    def dump_file(self, file_url, path=None):
        if path is None:
            self.set_dump_path()
            path = self.path
        


"""        self.path = os.path.join(os.path.dirname(__file__),
                                 '../cfg/config.yaml')"""
