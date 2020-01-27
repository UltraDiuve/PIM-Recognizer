"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import requests
import os
import json
import warnings
import pandas as pd
import numpy as np
import progressbar

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
        try:
            self._load_directory()
        except FileNotFoundError:
            warnings.warn('No directory found for current env. '
                          'A new directory should be set.')

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

    def fetch(self, iter_uid=None, from_='PIM', **kwargs):
        """Fetches data from an uid iterable, either from PIM or from disk

        If no uid iterable is provided, no filter is applied to the result.
        """
        if from_ == 'PIM':
            self.fetch_from_PIM(iter_uid, **kwargs)
        else:
            raise NotImplementedError(f'Unexpected from argument : {from_}')

    def fetch_from_PIM(self, iter_uid=None, nx_properties='*', **kwargs):
        """Fetches data from an uid iterable, from PIM
        """
        query = (f"SELECT * "
                 f"FROM Document "
                 f"WHERE ecm:primaryType='pomProduct' "
                 f"AND ecm:isVersion=0")
        if iter_uid is not None:
            uid_string = ', '.join(["'" + str(uid) + "'" for uid in iter_uid])
            query += f"AND ecm:uuid in ({uid_string})"
        headers = {'Content-Type': 'application/json',
                   'X-NXproperties': nx_properties}
        params = {'query': query}
        url = (self.cfg.baseurl +
               self.cfg.suffixid +
               self.cfg.rootuid + '/' +
               '@search')
        self.run_query(headers, params, url, **kwargs)

    def run_query(self, headers, params, url, threads=1, max_page=None,
                  page_size=None, **kwargs):
        if threads > 1:
            raise NotImplementedError('Multithreading not yet implemented')
        max_page = max_page if max_page else self.cfg.maxpage
        page_size = page_size if page_size else self.cfg.pagesize
        params['pageSize'] = page_size
        params['currentPageIndex'] = 0
        self.result = []
        self.result.append(requests.get(url,
                                        proxies=self.proxies,
                                        headers=headers,
                                        params=params,
                                        auth=(self.cfg.user,
                                              self.cfg.password)))
        resultsCount = self.result[0].json()['resultsCount']
        returnCount = resultsCount
        if max_page != -1 and resultsCount > (max_page * page_size):
            returnCount = max_page * page_size
            warnings.warn(f'\nMax size reached ! \n'
                          f'Only {returnCount} results will be '
                          f'fetched out of {resultsCount} results\n')
        bar_widgets = [progressbar.Counter(),
                       f'/{returnCount} ',
                       progressbar.Bar('#', '[', ']', '-')]
        bar = progressbar.ProgressBar(maxval=returnCount,
                                      widgets=bar_widgets,
                                      redirect_stdout=True,
                                      redirect_stderr=True)
        print(f'Running query with parameters max_page:{max_page}, page_size:'
              f'{page_size}')
        bar.start()
        while self.result[-1].json()['isNextPageAvailable']:
            bar.update(self.result[-1].json()['currentPageOffset'] + page_size)
            params['currentPageIndex'] += 1
            if max_page != -1 and params['currentPageIndex'] >= max_page:
                break
            self.result.append(requests.get(url,
                                            proxies=self.proxies,
                                            headers=headers,
                                            params=params,
                                            auth=(self.cfg.user,
                                                  self.cfg.password)))
        bar.finish()

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
        TODO : deprecate ce bullshit. check_if_fetch c'est bâtard et self.uid
        c'est de la merde.
        """
        self.check_if_fetched()
        self.path = os.path.join(self._root_path(),
                                 self.uid)

    def _create_folder(self, path):
        """Check if folder for dumping exists and creates is if needed

        Returns the path of the folder created. This function creates a folder
        equal to path if passed as parameter, or defaults its value to inner
        attribute self.path if not passed as parameter.
        TODO : deprecate avec une façon plus élégante de faire.
        """
        if path is None:
            self.check_if_fetched()
            self._set_dump_path()
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
        return(path)

    def dump_data_from_result(self, filename='data.json'):
        """Dumps data from result as JSON files

        This functions dumps data from result as JSON files.
        Note that result MUST be an iterable of responses (be it from PIM or
        from disk)
        TODO : make it update the directory
        TODO : TEST
        """
        for single_result in self.result:
            doc_list = single_result.json()['entries']
            for document in doc_list:
                path = os.path.join(self._root_path(), document['uid'])
                if not os.path.exists(path):
                    os.makedirs(path)
                full_path = os.path.join(path, filename)
                with open(full_path, 'w+') as outfile:
                    json.dump(document, outfile)

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
            try:
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
            except TypeError:
                pass

    def get_directory(self, **kwargs):
        """Get the uid directory for the environment as a pandas DataFrame

        This function fetches the uid directory data for the current
        environment as a pandas DataFrame.
        To fetch all the results, set max_page attribute to -1
        """
        self.fetch_from_PIM(nx_properties='', **kwargs)
        df_list = []
        for single_result in self.result:
            df_list.append(pd.DataFrame(single_result.json()['entries'])
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
        self._directory = self._format_as_directory(self._directory)

    def _init_as_directory(self, df):
        """Sets initial dates values for dataframes to be used as directories
        """
        df['lastRefreshed'] = pd.Timestamp.now(tz='UTC')
        df['lastFetchedData'] = np.nan
        df['lastFetchedFiles'] = np.nan
        return(self._format_as_directory(df))

    def _format_as_directory(self, df):
        types = {'lastFetchedData': 'datetime64[ns, UTC]',
                 'lastFetchedFiles': 'datetime64[ns, UTC]'}
        df = df.astype(types)
        df = df.loc[:, self._directory_headers()]
        return(df)

    def reset_directory(self, page_size=None, max_page=None, filename=None):
        """COMPLETELY RESETS the directory for the environment

        Warning: this function will completely reset the directory for the
        current environment. It should only be used when first setting the
        directory or if it requires being remade from scratch.
        Warning: this function should never be used with max_page parameter
        different from -1 as it could result in incomplete data to erase
        current directory. Only relevant usage is for debugging purpose.
        """
        self._directory = self._init_as_directory(
            self.get_directory(max_page=max_page, page_size=page_size))
        self._save_directory()

    def refresh_directory(self, max_page=None, filename=None, page_size=None):
        """Refreshes the directory for the environment

        Warning: this function should never be used with max_page parameter
        different from -1 except for debugging purpose. Yet, doing so does
        not corrupt or lose data whatsoever.
        """
        new_dir = self.get_directory(max_page=max_page, page_size=page_size)
        new_dir = self._init_as_directory(new_dir)
        self._load_directory(filename=filename)
        cur_dir = self._format_as_directory(self._directory)
        cur_dir.update(new_dir)
        cur_dir = pd.concat([cur_dir,
                             new_dir[~new_dir.index.isin(cur_dir.index)]])
        self._directory = cur_dir
        self._save_directory()

    def fetch_modified_files(self, batch_size=100, **kwargs):
        """Fetches modified files for uids considered modified according to dir

        TODO : cette manière de récupérer les données est bullshit... Nettoyer
        la façon de fetcher les résultat parce qu'avec batch_size c'est tout
        perrave.
        """
        self._load_directory()
        df = self._directory
        mask = (df.lastFetchedFiles.isna() |
                (df.lastModified > df.lastFetchedFiles))
        if len(mask.index) > batch_size:
            self.fetch_from_PIM(**kwargs)
        else:
            self.fetch_from_PIM(iter_uid=mask.index[:batch_size], **kwargs)
        return('pou')
        for uid in df[mask].index:
            print(f'Fetching files for {uid}')
            self.get_info_from_uid(uid)
            self.dump_attached_files()
            self._directory.loc[uid, 'lastFetchedFiles'] = (pd.Timestamp
                                                            .now(tz='UTC'))
        self._save_directory()
        print('Done!')

    def _save_directory(self, filename=None):
        directory_filename = filename if filename else self.cfg.uiddirectory
        full_path = os.path.join(self._root_path(), directory_filename)
        self._directory.to_csv(full_path, encoding='utf-8-sig')

    def fetch_modified_data(self):
        self._load_directory()
        df = self._directory
        mask = (df.lastFetchedData.isna() |
                (df.lastModified > df.lastFetchedData))
        for uid in df[mask].index:
            print(f'Fetching data for {uid}')
        # TODO ! End this dev

    def fetch_modified(self, what={'files'}):
        """Fetches the data and/or files from source

        This methods bases on current state of the directory. It fetches the
        data and/or file that have been modified after last fetch.
        """
        self._load_directory()
        if 'data' in what:
            self.fetch_modified_data()
        if 'files' in what:
            self.fetch_modified_files()
