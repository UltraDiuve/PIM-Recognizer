"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import requests
import os
import json
import warnings
import pandas as pd
import numpy as np

from . import conf

warnings.simplefilter('always', UserWarning)

def compute_extension(filename):
    """Computes the extension from a filename

    Returns the extension. If the filename has no "." (dot) in it, returns an
    empty string. If the computed extension has strictly more than 4
    characters, returns the empty string."""
    splitted = filename.split('.')
    if len(splitted) == 1:
        return('')
    elif len(splitted[-1]) > 4:
        return('')
    else:
        return(filename.split('.')[-1])


class Requester(object):
    """Requester class to retrieve information from PIM
    """
    def __init__(self, env, proxies='default'):
        self.cfg = conf.Config(env)
        if proxies == 'default':
            try:
                proxies = self.cfg.proxies
            except (AttributeError, KeyError):
                warnings.warn('No proxy conf found - defaulted to None')
                proxies = None
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

    def _root_path(self):
        """Returns the root path for the dumps
        """
        return(os.path.join(os.path.dirname(__file__),
                            '..',
                            'dumps',
                            self.cfg.env))

    def _set_dump_path(self):
        """Computes and sets inner dump path attribute

        This function set path attribute used for dumping data to disk. The
        path is <root>/dumps/<env>/<uid>/
        """
        self.check_if_fetched()
        self.path = os.path.join(self._root_path(),
                                 self.uid)

    def _create_folder(self, path):
        """Check if folder for dumping exists and creates is if needed

        Returns the path of the folder created. This function creates a folder
        equal to path if passed as parameter, or defaults its value to inner
        attribute self.path if not passed as parameter.
        """
        if path is None:
            self.check_if_fetched()
            self._set_dump_path()
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
        return(path)

    def dump_data(self, path=None, filename='data.json'):
        self.check_if_fetched()
        path = self._create_folder(path)
        full_path = os.path.join(path, filename)
        with open(full_path, 'w+') as outfile:
            json.dump(self.result.json(), outfile)

    def _dump_file(self, file_url, path=None, filename='file.pdf'):
        path = self._create_folder(path)
        self.resp = requests.get(file_url,
                                 proxies=self.proxies,
                                 auth=(self.cfg.user, self.cfg.password),
                                 stream=True)
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb')as outfile:
            outfile.write(self.resp.content)

    def dump_attached_files(self, path=None):
        self.check_if_fetched()
        path = self._create_folder(path)
        for filekind, filedef in self.cfg.filedefs.items():
            pointer = self.result.json()
            for node in filedef['nuxeopath']:
                pointer = pointer[node]
            nxfilename = pointer['name']
            url = (self.cfg.baseurl +
                   self.cfg.suffixfile +
                   self.cfg.nxrepo +
                   self.uid + '/' +
                   filedef['nuxeopath'][-1])
            ext = compute_extension(nxfilename)
            filename = filedef['dumpfilename'] + '.' + ext
            self._dump_file(url, filename=filename)

    def get_directory(self, max_page=None, page_size=None):
        """Get the uid directory for the environment as a pandas DataFrame

        This function fetches the uid directory data for the current
        environment as a pandas DataFrame.
        To fetch all the results, set max_page attribute to -1
        """
        max_page = max_page if max_page else self.cfg.maxpage
        page_size = page_size if page_size else self.cfg.pagesize

        query = ("SELECT * "
                 "FROM Document "
                 "WHERE ecm:primaryType='pomProduct' "
                 "AND ecm:isVersion=0")

        url = (self.cfg.baseurl +
               self.cfg.suffixid +
               self.cfg.rootuid + '/' +
               '@search')

        params = {'pageSize': page_size,
                  'query': query,
                  'currentPageIndex': 1}

        headers = {'Content-Type': 'application/json',
                   'X-NXproperties': '',
                   'X-NXRepository': 'default'}

        resp = requests.get(url,
                            proxies=self.proxies,
                            headers=headers,
                            params=params,
                            auth=(self.cfg.user, self.cfg.password))

        df_list = [pd.DataFrame(resp.json()['entries']).set_index('uid')]

        while resp.json()['isNextPageAvailable']:
            if max_page != -1 and params['currentPageIndex'] >= max_page:
                warnings.warn('Max results reached, not all result have been'
                              ' fetched.')
                break
            params['currentPageIndex'] += 1
            resp = requests.get(url,
                                proxies=self.proxies,
                                headers=headers,
                                params=params,
                                auth=(self.cfg.user, self.cfg.password))
            df_list.append(pd.DataFrame(resp.json()['entries'])
                           .set_index('uid'))

        df = pd.concat(df_list)
        df['lastModified'] = pd.to_datetime(df.loc[:, 'lastModified'])
        return(df)

    def _directory_headers(self):
        headers = ['type',
                   'title',
                   'lastModified',
                   'lastRefreshed',
                   'lastFetchedData',
                   'lastFetchedFiles']
        return(headers)

    def _load_directory(self, filename=None):
        """Loads directory from disk into the tool attribute
        """
        directory_filename = filename if filename else self.cfg.uiddirectory
        full_path = os.path.join(self._root_path(), directory_filename)
        self._directory = (pd.read_csv(full_path,
                                       encoding='utf-8-sig',
                                       parse_dates=['lastModified',
                                                    'lastRefreshed',
                                                    'lastFetchedData',
                                                    'lastFetchedFiles'])
                           .set_index('uid'))

    def reset_directory(self, page_size=None, max_page=None, filename=None):
        """COMPLETELY RESETS the directory for the environment

        Warning: this function will completely reset the directory for the
        current environment. It should only be used when first setting the
        directory or if it requires being remade from scratch.
        Warning: this function should never be used with max_page parameter
        different from -1 as it could result in incomplete data to erase
        current directory. Only relevant usage is for debugging purpose.
        """
        df = self.get_directory(max_page=max_page, page_size=page_size)
        df['lastRefreshed'] = pd.Timestamp.now(tz='Europe/Paris')
        df['lastFetchedData'] = np.nan
        df['lastFetchedFiles'] = np.nan
        df = df.loc[:, self._directory_headers()]
        directory_filename = filename if filename else self.cfg.uiddirectory
        full_path = os.path.join(self._root_path(), directory_filename)
        df.to_csv(full_path, encoding='utf-8-sig')

    def refresh_directory(self, max_page=None, filename=None, page_size=None):
        """Refreshes the directory for the environment

        Warning: this function should never be used with max_page parameter
        different from -1 except for debugging purpose. Yet, doing so does
        not corrupt or lose data whatsoever.
        """
        self._load_directory(filename=filename)
        new_dir = self.get_directory(max_page=max_page, page_size=page_size)
        new_dir['lastRefreshed'] = np.nan
        new_dir['lastFetchedData'] = np.nan
        new_dir['lastFetchedFiles'] = np.nan
        new_dir = new_dir.loc[:, self._directory_headers()]
        cur_dir = self._directory
        cur_dir.update(new_dir)
        cur_dir = pd.concat([cur_dir,
                             new_dir[~new_dir.index.isin(cur_dir.index)]])
        cur_dir['lastRefreshed'] = pd.Timestamp.now(tz='Europe/Paris')
        directory_filename = filename if filename else self.cfg.uiddirectory
        full_path = os.path.join(self._root_path(), directory_filename)
        cur_dir.to_csv(full_path, encoding='utf-8-sig')

    def fetch_modified_data(self):
        self._load_directory()
        df = self._directory
        mask = (df.lastFetchedData.isna() |
                (df.lastModified > df.lastFetchedData))
        return(df[mask])

    def fetch_modified(self, what={'data', 'files'}):
        """Fetches the data and/or files from source

        This methods bases on current state of the directory. It fetches the
        data and/or file that have been modified after last fetch.
        """
        self._load_directory()
        if 'data' in what:
            self.fetch_modified_data()
