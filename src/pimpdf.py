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


class PDFDecoder(object):
    """Tool that provides basic pdf decoding functionnalities
    """

    @staticmethod
    def path_to_text(path):
        """Decodes file at local path in the form of a long string
        """
        output_string = StringIO()
        with open(path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        return(output_string.getvalue())

    @staticmethod
    def path_to_blocks(path):
        """Decodes file at local path in the form of a list of blocks

        Blocks are part of the original string separated by at least 2
        carriage returns (i.e. with at least a single blank line between them)
        """
        return(PDFDecoder.path_to_text(path).split('\n\n'))

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
