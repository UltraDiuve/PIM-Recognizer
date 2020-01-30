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
import copy
import threading

from . import conf

warnings.simplefilter('always', UserWarning)


class Requester(object):
    """Requester class to retrieve information from PIM
    """
    def __init__(self, env, proxies='default'):
        self.cfg = conf.Config(env)
        self.session = requests.Session()
        self.session.auth = (self.cfg.user, self.cfg.password)
        if proxies == 'default':
            try:
                proxies = self.cfg.proxies
            except (AttributeError, KeyError):
                warnings.warn('No proxy conf found - defaulted to None')
                proxies = None
        self.proxies = proxies
        self.rlock = threading.RLock()
        self.result = []
        try:
            self._load_directory()
        except FileNotFoundError:
            warnings.warn('No directory found for current env. '
                          'A new directory should be set.')

    def fetch(self, iter_uid=None, from_='PIM', **kwargs):
        """Fetches data from an uid iterable, either from PIM or from disk

        If no uid iterable is provided, no filter is applied to the result.
        TODO : review this method :
            - should call fetch all or fetch list
            - add fetch from disk
        """
        if from_ == 'PIM':
            self.fetch_from_PIM(iter_uid, **kwargs)
        else:
            raise NotImplementedError(f"Unexpected 'from_' argument : {from_}")

    def fetch_all_from_PIM(self, nx_properties='*', max_page=None,
                           page_size=None,):
        """Fetches all product data from PIM into result

        This method fetches all product data from PIM. It implements
        multithreading to speed up retrieval.
        """
        query = (f"SELECT * "
                 f"FROM Document "
                 f"WHERE ecm:primaryType='pomProduct' "
                 f"AND ecm:isVersion=0")
        headers = {'Content-Type': 'application/json',
                   'X-NXproperties': nx_properties}
        params = {'query': query}
        url = (self.cfg.baseurl +
               self.cfg.suffixid +
               self.cfg.rootuid + '/' +
               '@search')
        result_count = self.query_size(headers, params, url)
        self.result = []
        max_page = max_page if max_page else self.cfg.maxpage
        page_size = page_size if page_size else self.cfg.pagesize
        params['pageSize'] = page_size
        thread_count = result_count // page_size + 1
        if max_page != - 1 and thread_count > max_page:
            thread_count = max_page
            warnings.warn(f'\nMax size reached ! \n'
                          f'Only {max_page * page_size} results will be '
                          f'fetched out of {result_count} results\n')
        threads = []
        for page_index in range(thread_count):
            t = threading.Thread(target=self.get_page_from_query,
                                 args=(url, headers, params, page_index))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('Done')

    def get_page_from_query(self, url, headers, params, currentPageIndex=0):
        """Fetches data for a single page of results

        This methods fetches data for a prepared query for a single page. It
        is used to implement multithreading for `fetch_all_from_PIM`
        """
        try:
            local_params = copy.deepcopy(params)
            local_params['currentPageIndex'] = currentPageIndex
            resp = self.session.get(url,
                                    proxies=self.proxies,
                                    headers=headers,
                                    params=local_params)
            with self.rlock:
                self.result.append(resp)
        except Exception as e:
            print('An error occured in this thread!')
            print(e)

    def fetch_list_from_PIM(self, iter_uid, batch_size=50, nx_properties='*'):
        """Fetches data from an uid iterable, from PIM

        This method fetches the data from PIM based on a iterable of uids.
        It creates threads to fetch the complete list, and bases on the
        Nuxeo @search API with a WHERE clause. It may therefore be less
        fast than a full fetch.
        Due to http limitations, it requires that no more than 50 uid be
        retrieved at a time in a single thread. Failing to enforce will result
        in incomplete responses an missed results.
        """
        if batch_size > 50:
            raise ValueError(f'batch_size needs to be < 50. '
                             f'Call with:{batch_size}')
        query = (f"SELECT * "
                 f"FROM Document "
                 f"WHERE ecm:primaryType='pomProduct' "
                 f"AND ecm:isVersion=0")
        headers = {'Content-Type': 'application/json',
                   'X-NXproperties': nx_properties}
        url = (self.cfg.baseurl +
               self.cfg.suffixid +
               self.cfg.rootuid + '/' +
               '@search')
        uid_lists = [iter_uid[i:i+batch_size]
                     for i in range(0, len(iter_uid), batch_size)]
        thread_list = []
        self.result = []
        for uid_list in uid_lists:
            uid_string = ', '.join(["'" + str(uid) + "'" for uid in uid_list])
            local_query = query + f" AND ecm:uuid in ({uid_string})"
            params = {'query': local_query}
            t = threading.Thread(target=self.get_page_from_query,
                                 args=(url, headers, params))
            t.start()
            thread_list.append(t)
        for thread in thread_list:
            thread.join()
        print('Done')

    def query_size(self, headers, params, url):
        """Runs a single result query to get the number of results

        This methods takes a query to be sent to Nuxeo in order to count the
        number of results.
        It then execute the query with page_size=1 and max_page=1 to fetch a
        single result, and extracts the total number of results from the
        response.
        """
        loc_headers = copy.deepcopy(headers)
        loc_headers['X-NXproperties'] = ''
        loc_params = copy.deepcopy(params)
        loc_params['pageSize'] = 1
        loc_params['currentPageIndex'] = 0
        resp = self.session.get(url,
                                proxies=self.proxies,
                                headers=headers,
                                params=params)
        return(resp.json()['resultsCount'])

    def _root_path(self):
        """Returns the root path for the dumps

        TODO : set as a static method
        """
        return(os.path.join(os.path.dirname(__file__),
                            '..',
                            'dumps',
                            self.cfg.env))

    def dump_data_from_result(self, filename='data.json'):
        """Dumps data from result attribute as JSON files

        This method dumps data from result as JSON files.
        Note that result MUST be an iterable of responses (be it from PIM or
        from disk), each response should have an 'entries' list of documents.
        """
        if not hasattr(self, '_directory'):
            self._load_directory()
        now = pd.Timestamp.now(tz='UTC')
        threads = []
        for single_result in self.result:
            t = threading.Thread(target=self.dump_data_from_single_result,
                                 args=(single_result, filename, now))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('Done !')

    def dump_data_from_single_result(self, single_result, filename, now):
        """Dumps data from a single result

        This method dumps data from a single result passed as an argument.
        It is used for multithreading the result list.
        """
        try:
            doc_list = single_result.json()['entries']
            s_list = []
            for document in doc_list:
                path = os.path.join(self._root_path(), document['uid'])
                if not os.path.exists(path):
                    os.makedirs(path)
                full_path = os.path.join(path, filename)
                with open(full_path, 'w+') as outfile:
                    json.dump(document, outfile)
                s_list.append(pd.Series(now,
                                        index=[document['uid']],
                                        name='lastFetchedData'))
            df = pd.concat(s_list, axis=0)
            with self.rlock:
                self._directory.update(df)
                self._save_directory()
        except Exception as e:
            print('An error occured in this thread!')
            print(e)

    def dump_files_from_result(self):
        """Dumps attached files from result items on disk

        This method dumps files from PIM on disk.
        It also updates the local directory with current datetime.
        Note that result MUST be an iterable of responses (be it from PIM or
        from disk), each response should have an 'entries' list of documents,
        and each document mention the files attached to it.
        Attached files definition MUST be set in the config.yaml file.
        """
        if not hasattr(self, '_directory'):
            self._load_directory()
        now = pd.Timestamp.now(tz='UTC')
        threads = []
        print(f'Launching {len(self.result)} threads.')
        for single_result in self.result:
            t = threading.Thread(target=self.dump_files_from_single_result,
                                 args=(single_result, now))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('Done!')

    def dump_files_from_single_result(self, single_result, now):
        """Dumps attached files from items on disk - for a single result

        This method dumps files from a single result into the disk. It is used
        for multithreading in `dump_files_from_result` method.
        """
        doc_list = single_result.json()['entries']
        s_list = []
        for document in doc_list:
            path = os.path.join(self._root_path(), document['uid'])
            if not os.path.exists(path):
                os.makedirs(path)
            self.dump_attached_files(document, path)
            s_list.append(pd.Series(now,
                                    index=[document['uid']],
                                    name='lastFetchedFiles'))
        df = pd.concat(s_list, axis=0)
        with self.rlock:
            self._directory.update(df)
            self._save_directory()
        print('Thread complete!')

    def _dump_file(self, file_url, path, filename='file'):
        """Dumps a file on fisk from its url"""
        resp = self.session.get(file_url,
                                proxies=self.proxies,
                                auth=(self.cfg.user, self.cfg.password),
                                stream=True)
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as outfile:
            outfile.write(resp.content)

    def dump_attached_files(self, document, path):
        """Dumps attached files from a nuxeo document

        This function fetches attached files from a nuxeo document (stored as
        a JSON).
        Files definitions are stored in the configuration file.
        """
        for filekind, filedef in self.cfg.filedefs.items():
            try:
                pointer = document
                for node in filedef['nuxeopath']:
                    pointer = pointer[node]
                nxfilename = pointer['name']
                url = (self.cfg.baseurl +
                       self.cfg.suffixfile +
                       self.cfg.nxrepo +
                       document['uid'] + '/' +
                       filedef['nuxeopath'][-1])
                ext = Requester.compute_extension(nxfilename)
                filename = filedef['dumpfilename'] + '.' + ext
                self._dump_file(url, path, filename=filename)
            except TypeError:
                pass

    def get_directory(self, **kwargs):
        """Get the uid directory for the environment as a pandas DataFrame

        This function fetches the uid directory data for the current
        environment as a pandas DataFrame.
        To fetch all the results, set max_page attribute to -1.
        This function returns data as is from PIM, and requires it to be
        formatted as a directory later on with `_init_as_directory` or
        `_format_as_directory` methods. (else, columns are not consistent with
        directory definition)
        """
        self.fetch_all_from_PIM(nx_properties='', **kwargs)
        df_list = []
        for single_result in self.result:
            df_list.append(pd.DataFrame(single_result.json()['entries'])
                           .set_index('uid'))
        df = pd.concat(df_list)
        df['lastModified'] = pd.to_datetime(df.loc[:, 'lastModified'])
        return(df)

    def _directory_headers(self):
        """Returns the directory headers

        TODO : set as staticmethod
        """
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

        TODO : set as staticmethod
        """
        df['lastRefreshed'] = pd.Timestamp.now(tz='UTC')
        df['lastFetchedData'] = np.nan
        df['lastFetchedFiles'] = np.nan
        return(self._format_as_directory(df))

    def _format_as_directory(self, df):
        """Formats a dataframe as a directory from dtypes point of view

        This method formats a dataframe columns as to be consistent with
        directory definition. It DOES NOT set initial values (see
        `_init_as_directory`).
        TODO : set as staticmethod
        """
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
        current directory. Only relevant usage of max_page != -1 is for
        debugging purpose.
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

    def _save_directory(self, filename=None):
        """Saves current directory to disk

        This methods saves the current directory to disk.
        """
        directory_filename = filename if filename else self.cfg.uiddirectory
        full_path = os.path.join(self._root_path(), directory_filename)
        self._directory.to_csv(full_path, encoding='utf-8-sig')

    def modified_items(self, what='any', max_results=None):
        """Returns modified items according to directory

        This methods compares the date of last modification stored in the
        directory to the date of last fetching of data or attached files.
        If what = 'any', all outdated items will be returned
        If what = 'data', it will return only items with outdated data
        If what = 'files', it will return only items with outdated files
        max_results enables to fetch only a limited count of results.
        This returns an uid list
        """
        if not hasattr(self, '_directory'):
            self._load_directory()
        df = self._directory
        if what == 'any':
            mask = (df.lastFetchedFiles.isna() |
                    (df.lastModified > df.lastFetchedFiles) |
                    df.lastFetchedData.isna() |
                    (df.lastModified > df.lastFetchedData))
        elif what == 'data':
            mask = (df.lastFetchedData.isna() |
                    (df.lastModified > df.lastFetchedData))
        elif what == 'files':
            mask = (df.lastFetchedFiles.isna() |
                    (df.lastModified > df.lastFetchedFiles))
        else:
            raise ValueError(f"Unexpected 'what' argument: {what}")
        uid_list = mask.index[mask].tolist()
        if max_results:
            uid_list = uid_list[:max_results]
        return(uid_list)

    def modification_report(self):
        """Prints some elements about current state

        This methods bases on current state of directory.
        """
        if not hasattr(self, '_directory'):
            self._load_directory()
        print(f'Number of items: {len(self._directory)}')
        print(f'Number of items with outdated data: '
              f'{len(self.modified_items(what="data"))}')
        print(f'Number of items with outdated files: '
              f'{len(self.modified_items(what="files"))}')

    @staticmethod
    def compute_extension(filename):
        """Computes the extension from a filename

        Returns the extension. If the filename has no "." (dot) in it, returns
        an empty string. If the computed extension has strictly more than 4
        characters, returns the empty string."""
        splitted = filename.split('.')
        if len(splitted) == 1:
            return('')
        elif len(splitted[-1]) > 4:
            return('')
        else:
            return(filename.split('.')[-1])
