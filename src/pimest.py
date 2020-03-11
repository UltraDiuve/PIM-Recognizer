"""PIM Estimator module

This modules enables to create an estimator to identify which block is the
ingredient list from an iterable of text blocks.
"""

import os
from io import BytesIO

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.utils.validation import check_is_fitted

from .pimapi import Requester
from .pimpdf import PDFDecoder
from .conf import Config


def raise_if_not_a_df(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f'This transformer expects a pandas Dataframe '
                        f'object. Got an object of type \'{type(X)}\' '
                        f'instead')


def check_target_exists(target_exists):
    if target_exists not in {'raise', 'ignore', 'overwrite'}:
        raise ValueError(f'target_exists parameter should be set to '
                         f'\'raise\' or \'ignore\' or \'to_nan\'. Got '
                         f'\'{target_exists}\' instead.')


def raise_if_target(X, target, target_exists):
    if target in X.columns and target_exists == 'raise':
        raise RuntimeError(f'Column \'{target}\' already exists in input'
                           f'DataFrame.')


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


class PathGetter(object):
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
                 filename_factory=lambda x: 'FTF.pdf'):
        self.cfg = Config(env)
        self.ground_truth_uids = ground_truth_uids
        self.train_set_path = train_set_path
        self.ground_truth_path = ground_truth_path
        self.path_factory = path_factory
        self.filename_factory = filename_factory

    def fit(self, X, y=None):
        """No fit is required for this class.
        """
        if not self.train_set_path:
            self.train_set_path = os.path.join(*self.cfg.trainsetpath)
        if not self.ground_truth_path:
            self.ground_truth_path = os.path.join(*self.cfg.groundtruthpath)
        self.fitted_ = True
        return(self)

    def transform(self, X):
        """Returns the paths for the uids"""
        check_is_fitted(self)
        if 'path' in X.columns:
            raise RuntimeError('The Dataframe already has a column named '
                               '\'path\'')
        df = X.copy()
        df['path'] = None
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
            df.loc[uid, 'path'] = path
        return(df)

    def fit_transform(self, X, y=None):
        return(self.fit(X).transform(X))


class ContentGetter(object):
    """Class that fetches the content of documents on disk

    This class fetches the data from documents on disk as bytes.
    It requires a dataframe with a path column"""
    def __init__(self, missing_file='raise', target_exists='raise'):
        self.missing_file = missing_file
        self.target_exists = target_exists

    def fit(self, X, y=None):
        raise_if_not_a_df(X)
        self.raise_if_no_path(X)
        raise_if_target(X, 'content', self.target_exists)
        self.raise_if_no_file(X)
        if self.missing_file not in {'raise', 'ignore', 'to_nan'}:
            raise ValueError(f'missing_file parameter should be set to '
                             f'\'raise\' or \'ignore\' or \'to_nan\'. Got '
                             f'\'{self.missing_file}\' instead.')
        check_target_exists(self.target_exists)
        self.fitted_ = True
        return(self)

    def raise_if_no_path(self, X):
        if 'path' not in X.columns:
            raise KeyError('Input DataFrame has no \'path\' column.')

    def raise_if_no_file(self, X):
        if self.missing_file == 'raise':
            mask = pd.DataFrame(index=X.index)
            mask['file_exists'] = X['path'].apply(ContentGetter.file_exists)
            if not mask['file_exists'].all():
                example_uid = mask.loc[~mask['file_exists']].index[0]
                example_path = X.loc[example_uid, 'path']
                raise RuntimeError(f'No file found for uid \'{example_uid}\' '
                                   f'at path \'{example_path}\'')

    def transform(self, X):
        check_is_fitted(self)
        raise_if_not_a_df(X)
        self.raise_if_no_path(X)
        raise_if_target(X, 'content', self.target_exists)
        X = X.copy()
        if 'content' in X.columns and self.target_exists == 'ignore':
            return(X)
        self.raise_if_no_file(X)
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

    def fit_transform(self, X, y=None):
        return(self.fit(X).transform(X))


class PDFContentParser(object):
    """Class that parses pdf content to text

    This class converts a file content (in the form of bytes) into text, using
    pimpdf functionalities (based on pdfminer.six)
    """
    def __init__(self, target_exists='raise'):
        self.target_exists = target_exists

    def raise_if_no_content(self, X):
        if 'content' not in X.columns:
            raise KeyError('This transformer requires a \'content\' column.'
                           ' None was found in the current DataFrame')

    def fit(self, X, y=None):
        raise_if_not_a_df(X)
        check_target_exists(self.target_exists)
        raise_if_target(X, 'text', self.target_exists)
        self.raise_if_no_content(X)
        self.fitted_ = True
        return(self)

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()
        X['text'] = PDFDecoder.threaded_contents_to_text(X['content'])
        return(X)

    def fit_transform(self, X, y=None):
        return(self.fit(X).transform(X))
