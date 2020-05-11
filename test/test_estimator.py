"""
Unit test for pimest module
"""

import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial

from numpy.linalg import norm

from sklearn.exceptions import NotFittedError

from src.pimest import PathGetter
from src.pimest import ContentGetter
from src.pimest import IngredientExtractor
from src.pimest import PIMIngredientExtractor
from src.pimest import PDFContentParser
from src.pimest import BlockSplitter
from src.pimest import SimilaritySelector
from src.pimest import DummyEstimator
from src.pimest import custom_accuracy
from src.pimest import text_sim_score


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
def df_path(gt_dataframe):
    df = gt_dataframe.copy()
    df['path'] = Path(__file__).parent / 'test_data' / '001_formatted_pdf.pdf'
    return(df)


@pytest.fixture
def df_incorrect_path(gt_dataframe):
    df = gt_dataframe.copy()
    df['path'] = Path('.', 'nosuchpath', 'nosuchfile.txt')
    return(df)


@pytest.fixture
def df_incorrect_path_with_content(df_incorrect_path):
    df = df_incorrect_path.copy()
    df['content'] = b'some data.\x00\x04'
    return(df)


@pytest.fixture
def df_ingredients():
    ingred_list = ['sucre, eau, légumes, farine de blé',
                   'malodextrine, colorant: E210, jus d\'orange',
                   '100% haricots blancs']
    idx = pd.Index(['001', '002', '003'], name='uid')
    df = pd.DataFrame(ingred_list, index=idx, columns=['Ingredients'])
    return(df)


@pytest.fixture
def df_candidates():
    blocks_list = ['Elaboré en union européenne, avec de la farine française',
                   'Sucre, haricots',
                   'Non soumis à TVA ni taxes diverses']
    idx = pd.Index(['001', '002', '003'], name='block_id')
    df = pd.DataFrame(blocks_list, index=idx, columns=['blocks'])
    return(df)


@pytest.fixture
def emphasized():
    beg, end = '\033[92m', '\033[0m'
    blocks_list = [f'Elaboré en union européenne, avec {beg}de{end} la '
                   f'{beg}farine{end} française',
                   f'{beg}Sucre{end}, {beg}haricots{end}',
                   'Non soumis à TVA ni taxes diverses']
    return(blocks_list)


@pytest.fixture
def df_content():
    path = Path(__file__).parent / 'test_data' / '001_formatted_pdf.pdf'
    with open(path, mode='rb') as file:
        content = file.read()
    contents = [content]
    idx = pd.Index(['001'], name='uid')
    df = pd.DataFrame(contents, index=idx, columns=['content'])
    return(df)


@pytest.fixture
def simil_df():
    ingred = ['eau, sucre',
              'malodextrine, farine de blé',
              '100% haricots blancs']
    blocks = [['Transformé en France', '100% sucre'],
              ['Les bons bonbons', 'agréement contact', 'E110, farine'],
              ['haricots', 'sans additif ni conservateur', 'Conforme']]
    index = pd.Index(['001', '002', '003'], name='uid')
    data = {'Ingrédients': ingred,
            'blocks': blocks}
    df = pd.DataFrame(data, index=index)
    return(df)


class TestDummy(object):
    def test_dummy(self):
        X = pd.DataFrame([[1, '2', 3], [4, '5', 6]],
                         columns=['A', 'B', 'C'])
        assert all(DummyEstimator().predict(X) == X)


class TestIngredientExtractor(object):
    def test_fit(self, df_ingredients):
        IngredientExtractor().fit(df_ingredients)

    def test_predict(self, df_ingredients, df_candidates):
        estimator = IngredientExtractor().fit(df_ingredients['Ingredients'])
        idx = estimator.predict(df_candidates['blocks'])
        assert idx == 1

    def test_show_emphasize(self, df_ingredients, df_candidates, emphasized):
        estimator = IngredientExtractor().fit(df_ingredients['Ingredients'])
        estimator.show_emphasize(df_candidates['blocks'])
        for idx, text in enumerate(df_candidates['blocks']):
            assert estimator.emphasize_words(text) == emphasized[idx]


class TestPIMIngredientExtractor(object):
    def test_passing(self, df_ingredients):
        extractor = PIMIngredientExtractor()
        extractor.fit(df_ingredients['Ingredients'])
        extractor.compare_uid_data('87b97662-c583-43fd-a1ed-d0b4f0eec54b')
        extractor.print_blocks()

    def test_invalid_uid(self, df_ingredients):
        extractor = PIMIngredientExtractor()
        extractor.fit(df_ingredients['Ingredients'])
        with pytest.raises(IndexError):
            extractor.compare_uid_data('not an uid')

    def test_unexpected_keyword(self):
        with pytest.raises(TypeError):
            PIMIngredientExtractor(incorrect_arg=True)

    def test_not_fitted(self):
        with pytest.raises(NotFittedError):
            (PIMIngredientExtractor()
             .compare_uid_data('87b97662-c583-43fd-a1ed-d0b4f0eec54b'))


class TestPathGetter(object):

    def test_init(self):
        PathGetter(env='dev')
        PathGetter(env='int')
        PathGetter(env='rec')
        PathGetter(env='qat')
        PathGetter(env='prd')

    def test_fit(self, gt_dataframe):
        PathGetter(env='prd').fit(gt_dataframe)
        # Checking incorrect environment
        with pytest.raises(ValueError):
            PathGetter(env='toto').fit(gt_dataframe)

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
        with pytest.raises(RuntimeError):
            transformer.fit(data)
        data = gt_dataframe.copy()
        transformer.fit(data)
        data['path'] = 'a path'
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

    def test_get_params(self):
        transformer = PathGetter()
        arg_count = transformer.__init__.__code__.co_argcount
        assert len(transformer.get_params()) == arg_count - 1

    def test_set_params(self):
        PathGetter().set_params(source_col='toto')


class TestContentGetter(object):

    def test_init(self):
        ContentGetter(missing_file='ignore', target_exists='raise')
        ContentGetter(missing_file='raise', target_exists='overwrite')
        ContentGetter(missing_file='to_nan', target_exists='ignore')
        ContentGetter(missing_file='incorrect input',
                      target_exists='incorrect input',
                      )

    def test_fit(self, df_path):
        ContentGetter(missing_file='ignore').fit(df_path)
        ContentGetter(missing_file='raise').fit(df_path)
        ContentGetter(missing_file='to_nan').fit(df_path)
        ContentGetter(target_exists='ignore').fit(df_path)
        ContentGetter(target_exists='raise').fit(df_path)
        ContentGetter(target_exists='overwrite').fit(df_path)
        with pytest.raises(ValueError):
            ContentGetter(missing_file='incorrect input').fit(df_path)
        with pytest.raises(ValueError):
            ContentGetter(target_exists='incorrect input').fit(df_path)

    def test_transform_not_a_df(self):
        # Testing behavior when argument is not a pandas dataframe
        with pytest.raises(TypeError):
            ContentGetter().fit_transform('a string')

    def test_fit_not_a_df(self):
        # Testing behavior when argument is not a pandas dataframe
        with pytest.raises(TypeError):
            ContentGetter().fit('a string')

    def test_fit_no_path(self, gt_dataframe):
        # Testing behavior when input has no 'path' column
        with pytest.raises(KeyError):
            ContentGetter().fit(gt_dataframe)

    def test_transform_no_path(self, gt_dataframe):
        # Testing behavior when input has no 'path' column
        with pytest.raises(KeyError):
            ContentGetter().fit_transform(gt_dataframe)

    def test_fit_no_file_raise(self, df_incorrect_path):
        with pytest.raises(RuntimeError):
            (ContentGetter(missing_file='raise')
             .fit(df_incorrect_path))

    def test_transform_no_file_raise(self, df_incorrect_path, df_path):
        transformer = ContentGetter(missing_file='raise').fit(df_path)
        with pytest.raises(RuntimeError):
            transformer.transform(df_incorrect_path)

    def test_transform_no_file_ignore(self, df_incorrect_path_with_content):
        data = (ContentGetter(missing_file='ignore',
                              target_exists='overwrite')
                .fit_transform(df_incorrect_path_with_content))
        assert (data['content']
                .equals(df_incorrect_path_with_content['content']))

    def test_transform_target_exists(self,
                                     df_incorrect_path_with_content,
                                     df_path):
        transformer = ContentGetter(missing_file='ignore',
                                    target_exists='raise')
        with pytest.raises(RuntimeError):
            transformer.fit(df_incorrect_path_with_content)
        transformer.fit(df_path)
        with pytest.raises(RuntimeError):
            transformer.transform(df_incorrect_path_with_content)

    def test_transform_target_ignore(self,
                                     df_incorrect_path_with_content):
        transformer = ContentGetter(missing_file='ignore',
                                    target_exists='ignore')
        data = transformer.fit_transform(df_incorrect_path_with_content)
        assert data.equals(df_incorrect_path_with_content)

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

    def test_get_params(self):
        transformer = ContentGetter()
        arg_count = transformer.__init__.__code__.co_argcount
        assert len(transformer.get_params()) == arg_count - 1

    def test_set_params(self):
        ContentGetter().set_params(source_col='toto')


class TestPDFContentParser(object):
    def test_init(self):
        PDFContentParser(target_exists='raise')

    def test_incorrect_param(self, df_incorrect_path_with_content):
        with pytest.raises(ValueError):
            (PDFContentParser(target_exists='incorrect input')
             .fit(df_incorrect_path_with_content))

    def test_missing_input_col(self, df_path):
        with pytest.raises(KeyError):
            PDFContentParser().fit(df_path)

    def test_fit_transform(self, df_content, test_data_001):
        transformer = PDFContentParser()
        data = transformer.fit_transform(df_content)
        target = Path(test_data_001.txt).read_text()
        target += '\x0c'
        assert (data['text'] == target).all()

    def test_get_params(self):
        transformer = PDFContentParser()
        arg_count = transformer.__init__.__code__.co_argcount
        assert len(transformer.get_params()) == arg_count - 1

    def test_set_params(self):
        PDFContentParser().set_params(source_col='toto')


class TestBlockSplitter(object):
    def test_incorrect_param(self, gt_dataframe):
        X = gt_dataframe.copy()
        X['text'] = 'a text'
        transformer = BlockSplitter(splitter_func=3)
        with pytest.raises(TypeError):
            transformer.fit(X)
        transformer = BlockSplitter()

    def test_override_column_name(self, gt_dataframe):
        X = gt_dataframe.copy()
        transformer = BlockSplitter(source_col='col')
        assert (transformer.fit_transform(X)['blocks'].iloc[0] == ['GT'])

    def test_passing(self, gt_dataframe):
        X = gt_dataframe.copy()
        X['text'] = 'a line<sep>another<sep>coucou'
        transformer = BlockSplitter(splitter_func=lambda x: x.split('<sep>'))
        target = ['a line', 'another', 'coucou']
        assert transformer.fit_transform(X)['blocks'].iloc[0] == target

    def test_get_params(self):
        transformer = BlockSplitter()
        arg_count = transformer.__init__.__code__.co_argcount
        assert len(transformer.get_params()) == arg_count - 1

    def test_set_params(self):
        BlockSplitter().set_params(source_col='toto')


class TestSimilaritySelector(object):
    def test_base(self, simil_df):
        transformer = SimilaritySelector()
        transformer.fit(simil_df['blocks'], simil_df['Ingrédients'])

    def test_predict(self, simil_df):
        transformer = SimilaritySelector().fit(simil_df['blocks'],
                                               simil_df['Ingrédients'])
        test_blocks = [['fabriqué en Italie',
                        'mélange de nougat',
                        'sucre, eau et betteraves']]
        assert (all(transformer.predict(test_blocks) ==
                pd.Series(['sucre, eau et betteraves'])))

    def test_predict_not_fitted(self):
        with pytest.raises(NotFittedError):
            SimilaritySelector().predict([['1', '2']])

    def test_transform(self, simil_df):
        out_ds = (SimilaritySelector().fit(simil_df['blocks'],
                                           simil_df['Ingrédients'])
                                      .predict(simil_df['blocks']))
        target_data = ['100% sucre',
                       'E110, farine',
                       'haricots']
        target_ds = pd.Series(target_data,
                              simil_df.index,
                              )
        assert pd.Series(out_ds).equals(target_ds)

    def test_empty_blocks(self, simil_df):
        X = simil_df.copy()
        X['blocks'].iloc[1] = ['']
        assert (SimilaritySelector().fit(X['blocks'],
                                         X['Ingrédients'])
                                    .predict(X['blocks'])[1] == '')
        model = SimilaritySelector().fit(X['blocks'],
                                         X['Ingrédients'])
        model.predict(X['blocks'].iloc[0])
        assert model.predict([['']]) == np.array([''])

    def test_empty_ingred(self, simil_df):
        X = simil_df.copy()
        X['Ingrédients'].iloc[1] = np.nan
        (SimilaritySelector().fit(X['blocks'], X['Ingrédients'])
                             .predict(X['blocks']))

    def test_predict_no_transform(self, simil_df):
        transformer = SimilaritySelector().fit(simil_df['blocks'],
                                               simil_df['Ingrédients'])
        assert (all(transformer.predict([['haricot', 'exploité en Inde']]) ==
                pd.Series(['haricot'])))

    def test_incorrect_param(self, simil_df):
        with pytest.raises(ValueError):
            (SimilaritySelector(similarity='incorrect input')
                .fit(simil_df['blocks'], simil_df['Ingrédients']))
        model = SimilaritySelector(similarity='projection',
                                   projected_norm='incorrect input')
        with pytest.raises(ValueError):
            model.fit(simil_df['blocks'], simil_df['Ingrédients'])

    def test_non_sparse_norm_type(self, simil_df):
        non_sparse_norm = partial(norm, axis=1, ord=1)
        model = SimilaritySelector(similarity='projection',
                                   projected_norm=non_sparse_norm,
                                   )
        with pytest.raises(ValueError):
            model.fit(simil_df['blocks'], simil_df['Ingrédients'])

    def test_count_vect_kwargs(self, simil_df):
        model = SimilaritySelector(count_vect_kwargs={'binary': True})
        model.fit(simil_df['blocks'], simil_df['Ingrédients'])
        model = SimilaritySelector(count_vect_kwargs={'incorrect': True})
        with pytest.raises(ValueError):
            model.fit(simil_df['blocks'], simil_df['Ingrédients'])
        model = (SimilaritySelector(
                 count_vect_kwargs={'strip_accents': 'incorrect'}))
        with pytest.raises(ValueError):
            model.fit(simil_df['blocks'], simil_df['Ingrédients'])

    def test_get_params(self):
        transformer = PathGetter()
        arg_count = transformer.__init__.__code__.co_argcount
        assert len(transformer.get_params()) == arg_count - 1


class TestAccuracy(object):

    def test_simple_accuracy(self):
        base_df = pd.DataFrame([['toto', 'toto']],
                               columns=['A', 'B'])
        passing1 = pd.DataFrame([['toto1-totoa', 'toto1-totoa']],
                                columns=['A', 'B'])
        test = pd.concat([base_df, passing1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=False,
                               lowercase=False,
                               strip_accents=None,
                               ) == 1.

        accent1 = pd.DataFrame([['tötôà', 'totoa']],
                               columns=['A', 'B'])
        test = pd.concat([base_df, accent1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=False,
                               lowercase=False,
                               strip_accents='unicode',
                               ) == 1.

        lowercase1 = pd.DataFrame([['TotOA', 'tOtoa']],
                                  columns=['A', 'B'])
        test = pd.concat([base_df, lowercase1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=False,
                               lowercase=True,
                               strip_accents=None,
                               ) == 1.

        whitespace1 = pd.DataFrame([['\ttotoa\n \t alt', 'totoa  alt ']],
                                   columns=['A', 'B'])
        test = pd.concat([base_df, whitespace1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=True,
                               lowercase=False,
                               strip_accents=None,
                               ) == 1.

        punctuation1 = pd.DataFrame([["j'en ai marre ! de toi.",
                                     'en ai marre de : toi...']],
                                    columns=['A', 'B'])
        test = pd.concat([base_df, punctuation1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=True,
                               lowercase=False,
                               strip_accents=None,
                               ) == 1.

        totale1 = pd.DataFrame([["J'en ài MA%CLAQUE !\n T'es nãze !!.\tzobi",
                                'en ai ma claque es naze zobi']],
                               columns=['A', 'B'])
        test = pd.concat([base_df, totale1], axis=0)
        assert custom_accuracy(DummyEstimator(),
                               test['A'],
                               test['B'],
                               tokenize=True,
                               lowercase=True,
                               strip_accents='unicode',
                               ) == 1.


class TestTextSimilarity(object):
    def test_text_similarity(self):
        # longest texts have a length of 20 after preprocessing
        text_df = pd.DataFrame([['arômes: Sucre   (E30 - E20)',
                                 'aromes, surce: E30 E20'],
                                [' émulsifiant : mon- et',
                                 'EMULSIFIANTS MONO ET']],
                               columns=['A', 'B'])
        assert text_sim_score(DummyEstimator(),
                              text_df['A'],
                              text_df['B'],
                              tokenize=True,
                              lowercase=True,
                              strip_accents='unicode',
                              similarity='levenshtein',
                              ) == 0.9
        assert text_sim_score(DummyEstimator(),
                              text_df['A'],
                              text_df['B'],
                              tokenize=True,
                              lowercase=True,
                              strip_accents='unicode',
                              similarity='damerau-levenshtein',
                              ) == 0.925
        text_sim_score(DummyEstimator(),
                       text_df['A'],
                       text_df['B'],
                       tokenize=True,
                       lowercase=True,
                       strip_accents='unicode',
                       similarity='jaro',
                       )
        text_sim_score(DummyEstimator(),
                       text_df['A'],
                       text_df['B'],
                       tokenize=True,
                       lowercase=True,
                       strip_accents='unicode',
                       similarity='jaro-winkler',
                       )

    def test_invalid_input(self):
        text_df = pd.DataFrame([['arômes: Sucre   (E30 - E20)',
                                 'aromes, surce: E30 E20'],
                                [' émulsifiant : mon- et',
                                 'EMULSIFIANTS MONO ET']],
                               columns=['A', 'B'])
        with pytest.raises(NotImplementedError):
            text_sim_score(DummyEstimator(),
                           text_df['A'],
                           text_df['B'],
                           tokenize=True,
                           lowercase=True,
                           strip_accents='unicode',
                           similarity='incorrect input',
                           ) == 0.9
