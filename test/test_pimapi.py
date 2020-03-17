"""
Unit test for pimapi module
"""

import pytest
import socket

from src.pimapi import Requester
from requests.exceptions import ConnectionError


class TestRequester(object):

    def test_environments(self, env):
        # Checking incorrect environment
        with pytest.raises(ValueError):
            Requester('toto')
        Requester(env)

    @pytest.mark.skipif(socket.gethostname() == 'devdm',
                        reason='connection always work when on devdm')
    def test_proxies(self):
        incorrect_proxies = {'http': 'http://incorrectproxy',
                             'https': 'https://incorrectproxy'}
        with pytest.raises(ConnectionError):
            Requester('prd', proxies=incorrect_proxies)

    def test_correct_credentials(self, env):
        assert Requester(env).check_credentials()

    def test_incorrect_credentials(self):
        incorrect_credentials = ('incorrect_user', 'incorrect_password')
        requester = Requester('prd', auth=incorrect_credentials)
        requester.check_connection()
        with pytest.raises(ConnectionError):
            requester.check_credentials()
    