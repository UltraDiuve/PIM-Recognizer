"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

import pandas as pd
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessPool as Pool
from io import StringIO, BytesIO
from functools import partial

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


class PDFDecoder(object):
    """Tool that provides basic pdf decoding functionnalities
    """

    @staticmethod
    def content_to_text(content):
        """Decodes the binary passed as argument

        content arg must be a Bytesio object.
        """
        output_string = StringIO()
        parser = PDFParser(content)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string,
                               laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
        return(output_string.getvalue())

    @staticmethod
    def path_to_text(path, missing_file='raise'):
        """Decodes file at local path in the form of a long string
        """
        if missing_file not in {'raise', 'ignore'}:
            raise ValueError(f'Unexpected value for missing_file parameter. '
                             f'Got {missing_file} but only \'raise\' or '
                             f'\'ignore\' are expected.')
        try:
            with open(path, mode='rb') as content:
                return(PDFDecoder.content_to_text(content))
        except FileNotFoundError:
            if missing_file == 'raise':
                raise
            if missing_file == 'ignore':
                print(f'File not found at path: {path}')
                return('')
        except Exception as e:
            print(e)
            print(f'An error happened at path: {path}')
            return('')

    @staticmethod
    def text_to_blocks(text, split_func=lambda x: x.split('\n\n')):
        """Splits text passed as an argument with splitter function

        split_func should be defined as to return a list like object.
        """
        return(split_func(text))

    @staticmethod
    def text_to_blocks_series(text, index=None,
                              split_func=lambda x: x.split('\n\n'),
                              ):
        """Splits text passed as an argument (using splitter func) to a Series

        The index arg is passed as provided to pd.Series constructor, or if
        it is a scalar it is broadcasted as a constant.
        """
        blocks_list = PDFDecoder.text_to_blocks(text,
                                                split_func=split_func,
                                                )
        try:
            return(pd.Series(blocks_list, index=index))
        except TypeError:
            index = [index] * len(blocks_list)
            return(pd.Series(blocks_list, index=index))

    @staticmethod
    def path_to_blocks(path, split_func=lambda x: x.split('\n\n'),
                       missing_file='raise'):
        """Decodes file at local path in the form of a list of blocks

        Blocks are part of the original string separated by at least 2
        carriage returns (i.e. with at least a single blank line between them)
        """
        text = PDFDecoder.path_to_text(path, missing_file=missing_file)
        return(PDFDecoder.text_to_blocks(text, split_func=split_func))

    @staticmethod
    def path_to_blocks_series(path,
                              index=None,
                              split_func=lambda x: x.split('\n\n'),
                              missing_file='raise'):
        """Decodes file at local path in the form a pd Series of blocks
        """
        text = PDFDecoder.path_to_text(path, missing_file=missing_file)
        return(PDFDecoder.text_to_blocks_series(text,
                                                split_func=split_func,
                                                index=index))

    @staticmethod
    def paths_to_blocks(path_series, split_func=lambda x: x.split('\n\n'),
                        missing_file='raise'):
        """Decodes files for each path in path list as a blocks Dataseries

        Blocks are part of the original string after the splitting function
        has been applied.
        The input must be a pandas dataseries of paths.
        The output is another pd Series, with the same indexes as the initial
        series (broadcasted to match the block count for each path)
        """
        ds_list = []
        for uid, path in path_series.items():
            ds = (PDFDecoder
                  .path_to_blocks_series(path,
                                         split_func=split_func,
                                         index=uid,
                                         missing_file=missing_file))
            ds_list.append(ds)
        return(pd.concat(ds_list, axis=0))

    @staticmethod
    def threaded_paths_to_blocks(path_series, processes=None,
                                 split_func=lambda x: x.split('\n\n'),
                                 missing_file='raise',
                                 ):
        """Threaded version of paths_to_blocks method

        It takes as input a series which index is the uid of the products,
        and the values are the path to the document.
        processes argument is the number of processes to launch. If omitted,
        it defaults to the number of cpu cores on the machine.
        """
        processer = partial(PDFDecoder.path_to_blocks_series,
                            split_func=split_func, missing_file=missing_file)
        processes = processes if processes else cpu_count()
        print(f'Launching {processes} processes.')
        with Pool(nodes=processes) as pool:
            ds_list = pool.map(processer,
                               path_series, path_series.index)
        return(pd.concat(ds_list, axis=0))

    @staticmethod
    def threaded_contents_to_text(content_series, processes=None,
                                  ):
        """Threaded version of content_to_text method

        It takes as input a series which index is the uid of the products,
        and the values are the content (in the form of bytes) of the
        documents.
        processes argument is the number of processes to launch. If omitted,
        it defaults to the number of cpu cores on the machine.
        """
        processer = PDFDecoder.content_to_text
        processes = processes if processes else cpu_count()
        print(f'Launching {processes} processes.')
        in_ds = content_series.apply(BytesIO)
        with Pool(nodes=processes) as pool:
            tuples = (list(in_ds.index),
                      pool.map(processer, in_ds))
        ds = pd.Series(tuples[1], index=tuples[0])
        return(ds)

    @staticmethod
    def threaded_texts_to_blocks(text_series, processes=None,
                                 split_func=lambda x: x.split('\n\n'),
                                 ):
        """Threaded version of text_to_blocks_series method

        It takes as input a series which index is the uid of the products,
        and the values are the content (in the form of bytes) of the
        documents..
        processes argument is the number of processes to launch. If omitted,
        it defaults to the number of cpu cores on the machine.
        """
        processer = partial(PDFDecoder.text_to_blocks_series,
                            split_func=split_func)
        processes = processes if processes else cpu_count()
        print(f'Launching {processes} processes.')
        ds_list = []
        with Pool(nodes=processes) as pool:
            ds_list = pool.map(processer, text_series, text_series.index)
        ds = pd.concat(ds_list, axis=0)
        return(ds)
