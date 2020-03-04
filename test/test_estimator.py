"""
Unit test for pimest module
"""

import pytest
import os
import pandas as pd
from pathlib import Path

from sklearn.exceptions import NotFittedError

from src.pimest import PathGetter
from src.pimest import ContentGetter


@pytest.fixture
def gt_dataframe():
    import pandas as pd
    ground_truth_uid = '0ea7c122-afea-429f-b61a-3f213946d558'
    ground_truth_index = pd.Index([ground_truth_uid],
                                  dtype='object',
                                  name='uid')
    ground_truth_df = pd.DataFrame(data={'col': 'GT'},
                                   index=ground_truth_index)
    return(ground_truth_df)


@pytest.fixture
def ts_dataframe():
    import pandas as pd
    train_set_uid = '00e8019a-99ba-459e-b8b4-8cef8cc046af'
    train_set_index = pd.Index([train_set_uid],
                               dtype='object',
                               name='uid')
    train_set_dataframe = pd.DataFrame(data={'col': 'TS'},
                                       index=train_set_index)
    return(train_set_dataframe)


@pytest.fixture
def tmp_file(tmp_path):
    filepath = tmp_path / 'file.txt'
    filepath.write_text('A content')
    return(filepath)


@pytest.fixture
def df_incorrect_path(gt_dataframe):
    df = gt_dataframe.copy()
    df['path'] = Path('.', 'nosuchpath', 'nosuchfile.txt')
    return(df)


@pytest.fixture
def df_incorrect_path_with_content(df_incorrect_path):
    df = df_incorrect_path.copy()
    df['content'] = 'some data: \x00\x01'
    return(df)


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

    def test_fit(self, gt_dataframe):
        PathGetter(env='prd').fit(gt_dataframe)

    def test_not_fitted(self, gt_dataframe):
        with pytest.raises(NotFittedError):
            PathGetter().transform(gt_dataframe)

    def test_transform(self, gt_dataframe, ts_dataframe):
        ground_truth_path = os.path.join('..', 'ground_truth')
        train_set_path = os.path.join('..', 'dumps', 'prd')
        transformer = PathGetter(env='prd',
                                 ground_truth_uids=gt_dataframe.index,
                                 ground_truth_path=ground_truth_path,
                                 train_set_path=train_set_path)
        data_dataframe = pd.concat([gt_dataframe, ts_dataframe], axis=0)
        transformer.fit(data_dataframe)
        transformed = transformer.transform(data_dataframe)
        gt_uid = data_dataframe.loc[data_dataframe['col'] == 'GT'].index[0]
        ts_uid = data_dataframe.loc[data_dataframe['col'] == 'TS'].index[0]
        assert (transformed.loc[gt_uid, 'path'] ==
                os.path.join(ground_truth_path, gt_uid, 'FTF.pdf'))
        assert (transformed.loc[ts_uid, 'path'] ==
                os.path.join(train_set_path, ts_uid, 'FTF.pdf'))

    def test_transform_already_path(self, gt_dataframe):
        data = gt_dataframe.copy()
        data['path'] = 'a path'
        transformer = PathGetter(env='prd')
        transformer.fit(data)
        with pytest.raises(RuntimeError):
            transformer.transform(data)

    def test_fit_transform(self, gt_dataframe, ts_dataframe):
        df = pd.concat([gt_dataframe, ts_dataframe], axis=0)
        gt_uids = list(gt_dataframe.index)
        transformer = PathGetter(env='prd', ground_truth_uids=gt_uids)
        transformer.fit_transform(df)

    def test_factories(self, gt_dataframe, ts_dataframe):
        df = pd.concat([gt_dataframe, ts_dataframe], axis=0)
        gt_uids = list(gt_dataframe.index)
        ts_uids = list(ts_dataframe.index)

        def path_factory(x):
            return('path')

        def filename_factory(x):
            return('.'.join([x, 'pdf']))
        transformer = PathGetter(env='prd',
                                 ground_truth_uids=gt_uids,
                                 ground_truth_path='gtpath',
                                 train_set_path='tspath',
                                 path_factory=path_factory,
                                 filename_factory=filename_factory,
                                 ).fit(df)
        transformed = transformer.transform(df)
        target_gt = os.path.join('gtpath', 'path', gt_uids[0] + '.pdf')
        target_ts = os.path.join('tspath', 'path', ts_uids[0] + '.pdf')
        assert transformed.loc[gt_uids[0], 'path'] == target_gt
        assert transformed.loc[ts_uids[0], 'path'] == target_ts


class TestContentGetter(object):

    def test_init(self):
        ContentGetter(missing_file='ignore', target_exists='raise')
        ContentGetter(missing_file='raise', target_exists='overwrite')
        ContentGetter(missing_file='to_nan', target_exists='ignore')
        ContentGetter(missing_file='incorrect input',
                      target_exists='incorrect input',
                      )

    def test_fit(self):
        ContentGetter(missing_file='ignore').fit()
        ContentGetter(missing_file='raise').fit()
        ContentGetter(missing_file='to_nan').fit()
        with pytest.raises(ValueError):
            ContentGetter(missing_file='incorrect input').fit()
        with pytest.raises(ValueError):
            ContentGetter(target_exists='incorrect input').fit()

    def test_transform_not_a_df(self):
        # Testing behavior when argument is not a pandas dataframe
        with pytest.raises(TypeError):
            ContentGetter().fit_transform('a string')

    def test_transform_no_path(self, gt_dataframe):
        # Testing behavior when input has no 'path' column
        with pytest.raises(KeyError):
            ContentGetter().fit_transform(gt_dataframe)

    def test_transform_no_file_raise(self, df_incorrect_path):
        with pytest.raises(RuntimeError):
            (ContentGetter(missing_file='raise')
             .fit_transform(df_incorrect_path))

    def test_transform_no_file_ignore(self, df_incorrect_path_with_content):
        data = (ContentGetter(missing_file='ignore',
                              target_exists='overwrite')
                .fit_transform(df_incorrect_path_with_content))
        assert (data['content']
                .equals(df_incorrect_path_with_content['content']))

    def test_transform_no_file_to_nan(self, df_incorrect_path_with_content):
        data = df_incorrect_path_with_content.copy()
        assert not data['content'].isnull().all()
        data = ContentGetter(missing_file='to_nan',
                             target_exists='overwrite').fit_transform(data)
        assert data['content'].isnull().all()

    def test_transform_not_fitted(self, gt_dataframe):
        with pytest.raises(NotFittedError):
            ContentGetter().transform(gt_dataframe)

    def test_transform(self, gt_dataframe, tmp_file):
        data = gt_dataframe.copy()
        data['path'] = tmp_file
        with open(tmp_file, mode='rb') as file:
            file_content = file.read()
        data = ContentGetter(missing_file='raise',
                             target_exists='raise').fit_transform(data)
        assert (data['content'] == file_content).all()
