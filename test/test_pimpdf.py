"""
Unit test for pimpdf module
"""

import pytest
from pathlib import Path
from collections import namedtuple
import pandas as pd
import re

from src.pimpdf import PDFDecoder


@pytest.fixture
def test_data_001():
    test_data = namedtuple('Test_data', ['pdf', 'txt', 'blocks'])
    pdf_path = Path(__file__).parent / 'test_data' / '001_formatted_pdf.pdf'
    txt_path = Path(__file__).parent / 'test_data' / '001_text.txt'
    block_path = Path(__file__).parent / 'test_data' / '001_blocks.csv'
    return(test_data(pdf_path, txt_path, block_path))


class TestPDFDecoder(object):

    def test_path_to_text(self, test_data_001):
        target = Path(test_data_001.txt).read_text()
        target = target + '\x0c'
        assert PDFDecoder.path_to_text(test_data_001.pdf) == target
        print(target)
        print(PDFDecoder.path_to_text(test_data_001.pdf))

    def test_path_to_text_no_file(self):
        assert PDFDecoder.path_to_text('incorect/path') == ''

    def test_path_to_blocks(self, test_data_001):
        target = pd.read_csv(test_data_001.blocks,
                             encoding='latin-1',
                             sep=';',
                             header=None,
                             names=['blocks'])
        form_feed_df = pd.DataFrame(['\x0c'], columns=['blocks'])
        target = target.append(form_feed_df, ignore_index=True)

        def splitter(x):
            return(re.split(r'\n *\n', x))

        blocks = PDFDecoder.path_to_blocks(test_data_001.pdf,
                                           split_func=splitter)
        df_blocks = pd.DataFrame(blocks, columns=['blocks'])
        assert target.equals(df_blocks)
