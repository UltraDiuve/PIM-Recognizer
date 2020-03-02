"""
Unit test for pimest module
"""

import pytest
import pandas as pd
import os

from src.pimest import PathGetter
from src.pimest import ContentGetter


class TestPathGetter(object):

    def test_init(self):
        # Checking incorrect environment
        with pytest.raises(ValueError):
            PathGetter(env='toto')
        PathGetter(env='dev')
        PathGetter(env='int')
        PathGetter(env='rec')
        PathGetter(env='qat')
        PathGetter(env='prd')

    def test_fit(self):
        PathGetter(env='prd').fit(pd.DataFrame())

    def test_transform(self):
        ground_truth_uid = '0ea7c122-afea-429f-b61a-3f213946d558'
        ground_truth_index = pd.Index([ground_truth_uid],
                                      dtype='object',
                                      name='uid')
        ground_truth_df = pd.DataFrame(data={'col': 'GT'},
                                       index=ground_truth_index)
        ground_truth_path = os.path.join('..', 'ground_truth')
        train_set_path = os.path.join('..', 'dumps', 'prd')
        transformer = PathGetter(env='prd',
                                 ground_truth_uids=ground_truth_df.index,
                                 ground_truth_path=ground_truth_path,
                                 train_set_path=train_set_path)
        train_set_uid = '00e8019a-99ba-459e-b8b4-8cef8cc046af'
        data_index = pd.Index([ground_truth_uid, train_set_uid],
                              dtype='object',
                              name='uid')
        data_dataframe = pd.DataFrame(data={'col': ['GT', 'TS']},
                                      index=data_index)
        transformed = transformer.transform(data_dataframe)
        assert (transformed.loc[ground_truth_uid, 'path'] ==
                os.path.join(ground_truth_path, ground_truth_uid))
        assert (transformed.loc[train_set_uid, 'path'] ==
                os.path.join(train_set_path, train_set_uid))

    def test_transform_already_path(self):
        uid = '0ea7c122-afea-429f-b61a-3f213946d558'
        uid_index = pd.Index([uid],
                             dtype='object',
                             name='uid')
        data = pd.DataFrame(data={'path': 'a path'},
                            index=uid_index)
        transformer = PathGetter(env='prd')
        with pytest.raises(RuntimeError):
            transformer.transform(data)


class TestContentGetter(object):

    def test_init(self):
        ContentGetter(errors='ignore')

    def test_fit(self):
        ContentGetter(errors='ignore').fit()
        with pytest.raises(ValueError):
            ContentGetter(errors='incorrect input').fit()
