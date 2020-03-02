"""PIM Estimator module

This modules enables to create an estimator to identify which block is the
ingredient list from an iterable of text blocks.
"""

import os
from io import BytesIO

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import norm as sparse_norm
from .pimapi import Requester
from .pimpdf import PDFDecoder
from .conf import Config


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
        print(f'Fetching data from PIM for uid {uid}...')
        self.requester.fetch_list_from_PIM([uid])
        ingredient_list = (self.requester.result[0]
                           .json()['entries'][0]['properties']
                           ['pprodc:ingredientsList'])
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
                                               proxies=self.requester.proxies,
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

    def print_blocks(self, resp):
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
                 ground_truth_path=None):
        self.cfg = Config(env)
        self.ground_truth_uids = ground_truth_uids
        if train_set_path:
            self.train_set_path = train_set_path
        else:
            print(self.cfg.trainsetpath)
            self.train_set_path = os.path.join(*self.cfg.trainsetpath)
        if ground_truth_path:
            self.ground_truth_path = ground_truth_path
        else:
            self.ground_truth_path = os.path.join(*self.cfg.groundtruthpath)

    def fit(self, X, y=None):
        """No fit is required for this class.
        """
        return(self)

    def transform(self, X):
        """Returns the paths for the uids"""
        if 'path' in X.columns:
            raise RuntimeError('The Dataframe already has a column named '
                               '\'path\'')
        df = X
        df['path'] = None
        for uid in X.index:
            if uid in self.ground_truth_uids:
                path = os.path.join(self.ground_truth_path, uid)
            else:
                path = os.path.join(self.train_set_path, uid)
            df.loc[uid, 'path'] = path
        return(df)


class ContentGetter(object):
    """Class that fetches the content of documents on disk

    This class fetches the data from documents on disk as BytesIO objects.
    It requires a dataframe with a path column"""
    def __init__(self, errors='raise'):
        self.errors = errors

    def fit(self, X=None):
        if self.errors not in {'raise', 'ignore'}:
            raise ValueError('errors parameter should be set to \'raise\' or '
                             f'\'ignore\'. Got \'{self.errors}\' instead.')
        return(self)
