"""PIM Estimator module

This modules enables to create an estimator to identify which block is the
ingredient list from an iterable of text blocks.
"""

import os
from io import BytesIO

import numpy as np
from numpy.linalg import norm as dense_norm
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import csr_matrix
import pandas as pd
from pathlib import Path
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
from jellyfish import damerau_levenshtein_distance
from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro as jaro_similarity
from Levenshtein import jaro_winkler as jaro_winkler_similarity

from .pimapi import Requester
from .pimpdf import PDFDecoder
from .conf import Config


class CustomTransformer(object):
    """Abstract class for custom transformers
    """
    def __init__(self, source_col, target_col, target_exists):
        self.source_col = source_col
        self.target_col = target_col
        self.target_exists = target_exists

    def raise_if_not_a_df(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'This transformer expects a pandas Dataframe '
                            f'object. Got an object of type \'{type(X)}\' '
                            f'instead')

    def check_target_exists(self):
        if self.target_exists not in {'raise', 'ignore', 'overwrite'}:
            raise ValueError(f'target_exists parameter should be set to '
                             f'\'raise\' or \'ignore\' or \'to_nan\'. Got '
                             f'\'{self.target_exists}\' instead.')

    def raise_if_target(self, X):
        if self.target_col in X.columns and self.target_exists == 'raise':
            raise RuntimeError(f'Column \'{self.target_col}\' already exists '
                               f'in input DataFrame.')

    def raise_if_no_source(self, X):
        if self.source_col and self.source_col not in X.columns:
            raise KeyError(f'Input DataFrame has no \'{self.source_col}\' '
                           f'column.')

    def fit(self, X, y=None):
        self.check_target_exists()
        self.raise_if_not_a_df(X)
        self.raise_if_target(X)
        self.raise_if_no_source(X)

    def transform(self, X):
        check_is_fitted(self)
        self.check_target_exists()
        self.raise_if_not_a_df(X)
        self.raise_if_target(X)
        self.raise_if_no_source(X)
        if self.target_col in X.columns and self.target_exists == 'ignore':
            return(X)

    def fit_transform(self, X, y=None):
        return(self.fit(X).transform(X))

    def get_params(self, deep=True):
        parms = dict()
        parms['source_col'] = self.source_col
        parms['target_col'] = self.target_col
        parms['target_exists'] = self.target_exists
        return(parms)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return(self)


class IngredientExtractor(object):
    """Estimator that identifies the most 'ingredient like' block from a list
    """
    def __init__(self):
        """Constructor method of ingredient extractor"""
        pass

    def fit(self, X, y=None):
        """Fitter method of ingredient extractor

        X is an iterable of ingredient lists in the form of strings
        y is just here for compatibility in sklearn pipeline usage
        """
        self._count_vect = CountVectorizer()
        self.vectorized_texts_ = self._count_vect.fit_transform(X)
        self.vocabulary_ = self._count_vect.vocabulary_
        self.mean_corpus_ = self.vectorized_texts_.mean(axis=0)
        return(self)

    def predict(self, X):
        """Predicter method of ingredient extractor

        X is a list of text blocks.
        This methods returns the index of the text block that is most likely
        to hold the ingredient list"""
        X_against_ingred_voc = self._count_vect.transform(X)
        X_norms = sparse_norm(CountVectorizer().fit_transform(X), axis=1)
        X_dot_ingred = np.array(X_against_ingred_voc.sum(axis=1)).squeeze()
        pseudo_cosine_sim = np.divide(X_dot_ingred,
                                      X_norms,
                                      out=np.zeros(X_norms.shape),
                                      where=X_norms != 0)
        self.similarity_ = pseudo_cosine_sim
        return(np.argmax(pseudo_cosine_sim))

    def show_emphasize(self, X):
        """Method that prints strings with words from vocabulary emphasized
        """
        for text in self.emphasize_texts(X):
            print(text)

    def emphasize_texts(self, X):
        """Method that returns strings with words from vocabulary emphasized

        This method shows how some candidates texts are projected on the
        vocabulary that has been provided or gotten from fitting.
        It is useful to see how different blocks compare.
        X argument is an iterable of block candidates.
        """
        check_is_fitted(self)
        preprocessor = self._count_vect.build_preprocessor()
        tokenizer = self._count_vect.build_tokenizer()
        vocabulary = self._count_vect.vocabulary_
        emphasized_texts = []
        for block in X:
            text = self.emphasize_words(block,
                                        preprocessor=preprocessor,
                                        tokenizer=tokenizer,
                                        vocabulary=vocabulary,
                                        )
            emphasized_texts.append(text)
        return(emphasized_texts)

    def emphasize_words(self,
                        text,
                        preprocessor=None,
                        tokenizer=None,
                        vocabulary=None,
                        ansi_color='\033[92m',  # green by default
                        ):
        """Method that returns a string with words emhasized

        This methods takes a string and returns a similar string with the words
        emphasized (with color markers)
        """
        check_is_fitted(self)
        ansi_end_block = '\033[0m'
        if not preprocessor:
            preprocessor = self._count_vect.build_preprocessor()
        if not tokenizer:
            tokenizer = self._count_vect.build_tokenizer()
        if not vocabulary:
            vocabulary = self._count_vect.vocabulary_
        preprocessed_text = preprocessor(text)
        tokenized_text = tokenizer(preprocessed_text)
        idx = 0
        emphasized_text = ''
        for token in tokenized_text:
            if token in vocabulary:
                while preprocessed_text[idx: idx + len(token)] != token:
                    emphasized_text += text[idx]
                    idx += 1
                emphasized_text += (ansi_color + text[idx: idx + len(token)] +
                                    ansi_end_block)
                idx += len(token)
        emphasized_text += text[idx:]
        return(emphasized_text)

    def score(self, X, y):
        """Scorer method of ingredient extractor estimator

        X is an iterable of ingredient lists in the form of string
        y is the target as the index of the correct block.
        """
        pass


class PIMIngredientExtractor(IngredientExtractor):
    """Wrapped estimator that directly extracts the ingredient list from uid
    """
    def __init__(self, env='prd', **kwargs):
        self.requester = Requester(env, **kwargs)
        super().__init__()

    def compare_uid_data(self, uid):
        check_is_fitted(self)
        print(f'Fetching data from PIM for uid {uid}...')
        self.requester.fetch_list_from_PIM([uid])
        try:
            ingredient_list = (self.requester.result[0]
                               .json()['entries'][0]['properties']
                               ['pprodc:ingredientsList'])
        except IndexError:
            raise IndexError(f'Fetching data with uid {uid} ')
        print('----------------------------------------------------------')
        print(f'Ingredient list from PIM is :\n\n{ingredient_list}')
        print('\n----------------------------------------------------------')
        print(f'Supplier technical datasheet from PIM for uid {uid} is:')
        nuxeo_path = (self.requester.cfg.filedefs['supplierdatasheet']
                      ['nuxeopath'])
        pointer = self.requester.result[0].json()['entries'][0]
        for node in nuxeo_path:
            pointer = pointer[node]
        file_url = pointer['data']
        print(file_url)
        print('----------------------------------------------------------')
        print(f'Downloading content of technical datasheet file...')
        self.resp = self.requester.session.get(file_url,
                                               stream=True)
        resp = self.resp
        print('Done!')
        print('----------------------------------------------------------')
        print(f'Parsing content of technical datasheet file...')
        blocks = (PDFDecoder.content_to_text(BytesIO(resp.content))
                  .split('\n\n'))
        idx = self.predict(blocks)
        print('Done!')
        print('----------------------------------------------------------')
        print(f'Ingredient list extracted from technical datasheet:\n')
        print(blocks[idx])
        print('\n----------------------------------------------------------')

    def print_blocks(self):
        blocks = (PDFDecoder.content_to_text(BytesIO(self.resp.content))
                  .split('\n\n'))
        for i, block in enumerate(blocks):
            print(i, ' | ', block, '\n')


class PathGetter(CustomTransformer):
    """Class that gets path for documents on disk

    This class aims to compute the path to documents, in order to
    fetch documents from the correct folder (depending on whether
    they are from train set or from ground truth)
    All these can be set at initialization, if such is not the case
    then their values is gotten from the configuration file.
    """
    def __init__(self,
                 env='prd',
                 ground_truth_uids=None,
                 train_set_path=None,
                 ground_truth_path=None,
                 path_factory=lambda x: x,
                 filename_factory=lambda x: 'FTF.pdf',
                 source_col=None,
                 target_col='path',
                 target_exists='raise'
                 ):
        self.env = env
        self.ground_truth_uids = ground_truth_uids
        self.train_set_path = train_set_path
        self.ground_truth_path = ground_truth_path
        self.path_factory = path_factory
        self.filename_factory = filename_factory
        super().__init__(source_col=source_col, target_col=target_col,
                         target_exists=target_exists)

    def fit(self, X, y=None):
        """No fit is required for this class.
        """
        super().fit(X)
        self.cfg = Config(self.env)
        if not self.train_set_path:
            self.train_set_path = os.path.join(*self.cfg.trainsetpath)
        if not self.ground_truth_path:
            self.ground_truth_path = os.path.join(*self.cfg.groundtruthpath)
        self.fitted_ = True
        return(self)

    def transform(self, X):
        """Returns the paths for the uids"""
        super().transform(X)
        df = X.copy()
        df[self.target_col] = None
        for uid in X.index:
            if uid in self.ground_truth_uids:
                path = os.path.join(self.ground_truth_path,
                                    self.path_factory(uid),
                                    self.filename_factory(uid),
                                    )
            else:
                path = os.path.join(self.train_set_path,
                                    self.path_factory(uid),
                                    self.filename_factory(uid),
                                    )
            df.loc[uid, self.target_col] = path
        return(df)

    def get_params(self, deep=True):
        parms = super().get_params()
        parms['env'] = self.env
        parms['ground_truth_uids'] = self.ground_truth_uids
        parms['train_set_path'] = self.train_set_path
        parms['ground_truth_path'] = self.ground_truth_path
        parms['path_factory'] = self.path_factory
        parms['filename_factory'] = self.filename_factory
        return(parms)


class ContentGetter(CustomTransformer):
    """Class that fetches the content of documents on disk

    This class fetches the data from documents on disk as bytes.
    It requires a dataframe with a path column"""
    def __init__(self,
                 missing_file='raise',
                 target_exists='raise',
                 source_col='path',
                 target_col='content',
                 ):
        self.missing_file = missing_file
        super().__init__(source_col=source_col,
                         target_col=target_col,
                         target_exists=target_exists,
                         )

    def fit(self, X, y=None):
        super().fit(X)
        if self.missing_file not in {'raise', 'ignore', 'to_nan'}:
            raise ValueError(f'missing_file parameter should be set to '
                             f'\'raise\' or \'ignore\' or \'to_nan\'. Got '
                             f'\'{self.missing_file}\' instead.')
        self._raise_if_no_file(X)
        self.fitted_ = True
        return(self)

    def _raise_if_no_file(self, X):
        if self.missing_file == 'raise':
            mask = pd.DataFrame(index=X.index)
            mask['file_exists'] = X['path'].apply(ContentGetter.file_exists)
            if not mask['file_exists'].all():
                example_uid = mask.loc[~mask['file_exists']].index[0]
                example_path = X.loc[example_uid, 'path']
                raise RuntimeError(f'No file found for uid \'{example_uid}\' '
                                   f'at path \'{example_path}\'')

    def transform(self, X):
        super().transform(X)
        self._raise_if_no_file(X)
        X = X.copy()
        mask = pd.DataFrame(index=X.index)
        mask['file_exists'] = X['path'].apply(ContentGetter.file_exists)
        mask['target'] = X['path'].apply(ContentGetter.read_to_bytes)
        if self.missing_file == 'to_nan':
            idx_to_update = mask.index
        else:
            idx_to_update = mask['file_exists']
        X.loc[idx_to_update, 'content'] = mask.loc[idx_to_update, 'target']
        return(X)

    @staticmethod
    def read_to_bytes(path):
        try:
            return(Path(path).read_bytes())
        except FileNotFoundError:
            return(None)

    @staticmethod
    def file_exists(path):
        path = Path(path)
        return(path.is_file())

    def get_params(self, deep=True):
        parms = super().get_params()
        parms['missing_file'] = self.missing_file
        return(parms)


class PDFContentParser(CustomTransformer):
    """Class that parses pdf content to text

    This class converts a file content (in the form of bytes) into text, using
    pimpdf functionalities (based on pdfminer.six)
    """
    def __init__(self,
                 target_exists='raise',
                 source_col='content',
                 target_col='text',
                 none_content='raise'
                 ):
        self.none_content = none_content
        super().__init__(source_col=source_col,
                         target_col=target_col,
                         target_exists=target_exists,
                         )

    def fit(self, X, y=None):
        super().fit(X)
        self.fitted_ = True
        return(self)

    def transform(self, X):
        super().transform(X)
        X = X.copy()
        tran = (PDFDecoder
                .threaded_contents_to_text(X[self.source_col],
                                           none_content=self.none_content))
        X[self.target_col] = tran
        return(X)

    def get_params(self, deep=True):
        parms = super().get_params()
        parms['none_content'] = self.none_content
        return(parms)


class BlockSplitter(CustomTransformer):
    """Class that splits texts into blocks

    This class converts a text string into blocks (a list of string), using
    the splitter function provided
    """
    def __init__(self,
                 target_exists='raise',
                 source_col='text',
                 target_col='blocks',
                 splitter_func=(lambda x: x.split('\n\n'))
                 ):
        self.splitter_func = splitter_func
        super().__init__(target_exists=target_exists,
                         source_col=source_col,
                         target_col=target_col,
                         )

    def fit(self, X, y=None):
        super().fit(X)
        self._check_splitter_callable()
        self.fitted_ = True
        return(self)

    def _check_splitter_callable(self):
        self.splitter_func('')

    def transform(self, X):
        super().transform(X)
        X = X.copy()
        blocks = (PDFDecoder
                  .threaded_texts_to_blocks(X[self.source_col],
                                            split_func=self.splitter_func,
                                            return_type='as_list',
                                            ))
        X[self.target_col] = blocks
        return(X)

    def get_params(self, deep=True):
        parms = super().get_params()
        parms['splitter_func'] = self.splitter_func
        return(parms)


class SimilaritySelector():
    """Class that select the most similar block from a block list

    This class provides functionnalities to fit an estimator on a topic
    specific vocabulary, and to retrieve the best candidate amongst these
    blocks.
    """
    def __init__(self,
                 count_vect_kwargs=dict(),
                 similarity='projection',
                 source_norm='l2',
                 projected_norm='l1',
                 ):
        self.count_vect_kwargs = count_vect_kwargs
        self.similarity = similarity
        self.source_norm = source_norm
        self.projected_norm = projected_norm

    def get_params(self, deep=True):
        parms = dict()
        parms['count_vect_kwargs'] = self.count_vect_kwargs
        parms['similarity'] = self.similarity
        parms['source_norm'] = self.source_norm
        parms['projected_norm'] = self.projected_norm
        return(parms)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return(self)

    def fit(self, X, y):
        self._validate_similarity()
        self._validate_norms()
        # if norm not specified in count_vect_kwargs, set to None
        # and do not use idf by default
        if 'norm' not in self.count_vect_kwargs:
            self.count_vect_kwargs['norm'] = None
        if 'use_idf' not in self.count_vect_kwargs:
            self.count_vect_kwargs['use_idf'] = False
        try:
            self.count_vect = TfidfVectorizer(**self.count_vect_kwargs)
        except (TypeError):
            raise ValueError('Unexpected argument at init in '
                             'count_vect_kwargs.')
            raise
        try:
            self.count_vect.fit(y.fillna(''))
        except (ValueError):
            raise ValueError('Unexpected argument at fit in '
                             'count_vect_kwargs.')
            raise
        self.source_count_vect = TfidfVectorizer(**self.count_vect_kwargs)
        self._y_ = y.copy()
        self.fitted_ = True
        return(self)

    def _validate_similarity(self):
        if self.similarity not in {'projection', 'cosine'}:
            raise ValueError(f'similarity parameter should be set to '
                             f'\'projection\' or \'cosine\'. Got '
                             f'\'{self.similarity}\' instead.')

    def _validate_norms(self):
        norm_dict = {'l1': partial(sparse_norm, axis=1, ord=1),
                     'l2': partial(sparse_norm, axis=1, ord=2)}
        try:
            self.source_norm = norm_dict[self.source_norm]
        except KeyError:
            pass
        try:
            self.projected_norm = norm_dict[self.projected_norm]
        except KeyError:
            pass
        test_mat = csr_matrix([[0, 1], [2, 3]])
        try:
            self.projected_norm(test_mat)
        except (TypeError, ValueError) as e:
            print('Incorrect projected norm provided, see full stack for '
                  'details')
            raise ValueError(e)
        try:
            self.source_norm(test_mat)
        except (TypeError, ValueError) as e:
            print('Incorrect source norm provided, see full stack for '
                  'details')
            raise ValueError(e)

    def predict(self, X):
        """ function to predict best candidate

        X : a pandas Series of block lists, or a list of block lists.
        """
        check_is_fitted(self)
        docs = [text for block_list in X for text in block_list]
        try:
            self.source_count_vect.fit(docs)
        except ValueError:
            print('No words in blocks. Return defaulted to ""')
            return(np.array([''] * len(X)))
        self.computed_sims_ = []
        predicted_texts = []
        if self.similarity == 'projection':
            for block_list in X:
                texts = self.source_count_vect.transform(block_list)
                # project texts on corpus space
                projected_texts = self.count_vect.transform(block_list)
                # Compute norm of source texts
                texts_norms = self.source_norm(texts)
                # Compute norm of projected texts
                projected_norms = self.projected_norm(projected_texts)
                sim = np.divide(projected_norms,
                                texts_norms,
                                out=np.zeros(texts_norms.shape),
                                where=texts_norms != 0)
                self.computed_sims_.append(sim)
                predicted_texts.append(block_list[np.argmax(sim)])
            if isinstance(X, pd.Series):
                return(pd.Series(predicted_texts, index=X.index))
            else:  # for example, X is a list
                return(pd.Series(predicted_texts))
        if self.similarity == 'cosine':
            # compute target vector in docs corpus space
            target_vector = self.source_count_vect.transform(self._y_)
            target_vector = np.asarray(target_vector.mean(axis=0)).ravel()
            # normalize target vector
            try:
                target_vector /= dense_norm(target_vector, ord=2)
            except ZeroDivisionError:
                pass
            print(target_vector)
            for block_list in X:
                candidates = self.source_count_vect.transform(block_list)
                # normalize candidates
                normalize(candidates, norm='l2', axis=1, copy=False)
                sim = np.dot(candidates.toarray(), target_vector)
                self.computed_sims_.append(sim)
                predicted_texts.append(block_list[np.argmax(sim)])
            if isinstance(X, pd.Series):
                return(pd.Series(predicted_texts, index=X.index))
            else:  # for example, X is a list
                return(pd.Series(predicted_texts))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return(self.predict(X))


class DummyEstimator(object):
    """Dummy estimator that predicts y as exactly X

    This estimator has been developped for testing purposes, as pytest
    does not support yet class fixtures.
    """
    def fit(self, X, y=None):
        self.fitted_ = True

    def predict(self, X):
        return(X.copy())


def build_text_processor(tokenize=True,
                         lowercase=True,
                         strip_accents='unicode',
                         **kwargs,
                         ):
    """ Generates a text preprocessor from sklearn CountVectorizer tools

    It is based on sklearn CountVectorizer functionalities.
    tokenize means that the input string will be tokenized as words before
    being glued back with single spaces. Its purpose is to handle
    whitespaces (newlines, tabs, multiple spaces, ...) and punctuation.
    kwargs are directly passed to CountVectorizer constructor, and will
    serve to process the texts. Most useful args are 'strip_accent' and
    'lowercase'.
    """
    preprocessor_countvect = CountVectorizer(lowercase=lowercase,
                                             strip_accents=strip_accents,
                                             **kwargs,
                                             )
    preprocessor = preprocessor_countvect.build_preprocessor()
    tokenizer = preprocessor_countvect.build_tokenizer()
    if tokenize:
        def transformer(x):
            return(' '.join(tokenizer(preprocessor(x))))
    else:
        transformer = preprocessor
    return(transformer)


def custom_accuracy(estimator, X, y, **kwargs):
    """ Scorer that computes accuracy of estimator, for strings

    This function enables to score an estimator that returns long texts,
    with some text processing.
    It computes an accuracy after text processing.
    See build_text_processor for information on arguments.
    """
    transformer = build_text_processor(**kwargs)

    y_pred = pd.Series(estimator.predict(X)).apply(transformer)
    return((y_pred == y.apply(transformer)).mean())


def text_similarity(a, b, *, similarity, jw_prefix_weight=0.1):
    """ Function that computes similarity of texts with selected method

    Similarity can be : 'levenshtein', 'damerau-levenshtein', 'jaro',
    'jaro-winkler'.
    In the case of jaro-winkler, a prefix weight can be passed
    """
    if not similarity:
        raise ValueError('similarity is a mandatory')
    if similarity == 'levenshtein':
        dist = levenshtein_distance(a, b)
    elif similarity == 'damerau-levenshtein':
        dist = damerau_levenshtein_distance(a, b)
    elif similarity == 'jaro':
        return(jaro_similarity(a, b))
    elif similarity == 'jaro-winkler':
        return(jaro_winkler_similarity(a, b, jw_prefix_weight))
    else:
        raise NotImplementedError(f'similarity set to {similarity}. This '
                                  f'method has not been implemented yet')
    try:
        return(1 - (dist / max(len(a), len(b))))
    except ZeroDivisionError:
        return(1.)


def text_sim_score(estimator,
                   X,
                   y,
                   *,
                   similarity,
                   tokenize=True,
                   jw_prefix_weight=0.1,
                   **kwargs,
                   ):
    """ Scorer that computes mean similarity for an estimator

    This function enables to score an estimator that returns long texts,
    with some text processing, by computing a mean silimarity between
    predicted and target texts.
    It computes this similarity after text processing.
    See build_text_processor for information on arguments.
    """
    transformer = build_text_processor(tokenize, **kwargs)
    y_pred = pd.Series(estimator.predict(X))
    y_trans, y_pred_trans = y.apply(transformer), y_pred.apply(transformer)
    df = pd.concat([y_trans, y_pred_trans], axis=1)
    similarity = partial(text_similarity,
                         similarity=similarity,
                         jw_prefix_weight=jw_prefix_weight,
                         )
    similarities = df.apply(lambda x: similarity(x.iloc[0], x.iloc[1]), axis=1)
    return(similarities.mean())
