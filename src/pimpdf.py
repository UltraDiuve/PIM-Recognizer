"""PIM API Module

This module aims to enable to fetch data from PIM system, into local folders.
"""

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import pandas as pd
from multiprocessing import Pool, cpu_count


class PDFDecoder(object):
    """Tool that provides basic pdf decoding functionnalities
    """

    @staticmethod
    def content_to_text(content):
        """Decodes the binary passed as argument
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
    def path_to_text(path):
        """Decodes file at local path in the form of a long string
        """
        try:
            with open(path, mode='rb') as content:
                return(PDFDecoder.content_to_text(content))
        except Exception as e:
            print(e)
            print(f'Could not read file at path: {path}')
            return('')

    @staticmethod
    def path_to_blocks(path, split_func=lambda x: x.split('\n\n')):
        """Decodes file at local path in the form of a list of blocks

        Blocks are part of the original string separated by at least 2
        carriage returns (i.e. with at least a single blank line between them)
        """
        return(split_func(PDFDecoder.path_to_text(path)))

    @staticmethod
    def paths_to_blocks(path_series):
        """Decodes files for each path in path list as a blocks Dataseries

        Blocks are part of the original string separated by at least 2
        carriage returns (i.e. with at least a single blank line between them)
        The path list must be a pandas dataseries ; the return is another
        dataseries, with the same indexes as the initial series.
        """
        ds_list = []
        for uid, path in path_series.items():
            try:
                blocks = PDFDecoder.path_to_blocks(path)
                index = [uid] * len(blocks)
                ds_list.append(pd.Series(blocks, index=index))
            except FileNotFoundError:
                pass
        return(pd.concat(ds_list, axis=0))

    @staticmethod
    def threaded_paths_to_blocks(path_series, processes=None):
        """Threaded version of paths_to_blocks method

        It takes as input a series which index is the uid of the products,
        and the values are the path to the document.
        processes argument is the number of processes to launch. If omitted,
        it defaults to the number of cpu cores on the machine.
        """
        processes = processes if processes else cpu_count()
        print(f'Launching {processes} processes.')
        with Pool(processes=processes) as pool:
            ds_list = pool.starmap(PDFDecoder.single_path_to_blocks,
                                   list(zip(path_series.index, path_series)))
        return(pd.concat(ds_list, axis=0))

    @staticmethod
    def single_path_to_blocks(uid, path):
        """Returns a series of text blocks in the document with path `path`

        This method is used by threaded_path_to_blocks to enable
        multiprocessing.
        It takes as arguments an uid of a product (which is given back as the
        return series repeated index) and returns a series whose values are
        the identified text blocks in the document at path `path`
        """
        values = PDFDecoder.path_to_blocks(path)
        index = [uid] * len(values)
        ds = pd.Series(values, index=index)
        return(ds)
