"""
Unit test for pimpdf module
"""

import pytest
from pathlib import Path
from collections import namedtuple

from src.pimpdf import PDFDecoder


@pytest.fixture
def test_data_001():
    test_data = namedtuple('Test_data', ['pdf', 'txt'])
    pdf_path = Path(__file__).parent / 'test_data' / '001_formatted_pdf.pdf'
    txt_path = Path(__file__).parent / 'test_data' / '001_text.txt'
    return(test_data(pdf_path, txt_path))


class TestPDFDecoder(object):

    def test_path_to_text(self, test_data_001):
        target = Path(test_data_001.txt).read_text()
        target = target + '\x0c'
        assert PDFDecoder.path_to_text(test_data_001.pdf) == target
        print(target)
        print(PDFDecoder.path_to_text(test_data_001.pdf))
