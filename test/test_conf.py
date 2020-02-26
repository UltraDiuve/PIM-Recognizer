"""
Unit test for conf module
"""

import pytest

from src.conf import Config


class TestConf(object):

    def test_conf_passing(self):
        cfg = Config('prd')
        assert cfg.env == 'prd'
        assert cfg.baseurl == 'https://produits.groupe-pomona.fr/'
        for env in {'dev', 'int', 'rec', 'qat'}:
            cfg = Config(env)
            assert hasattr(cfg, 'suffixid')
            assert hasattr(cfg, 'suffixfile')
            assert hasattr(cfg, 'nxrepo')
            assert hasattr(cfg, 'uiddirectory')
            assert hasattr(cfg, 'rootuid')
            assert hasattr(cfg, 'maxpage')
            assert hasattr(cfg, 'pagesize')
            assert hasattr(cfg, 'filedefs')
            assert hasattr(cfg, 'proxies')
            assert hasattr(cfg, 'user')
            assert hasattr(cfg, 'password')

    def test_conf_invalid_env(self):
        with pytest.raises(ValueError):
            Config('toto')

    def test_conf_invalid_attr(self):
        with pytest.raises(AttributeError):
            Config('prd').unknown_attr
