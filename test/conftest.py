import pytest
from pathlib import Path
from collections import namedtuple


@pytest.fixture
def test_data_001():
    test_data = namedtuple('Test_data', ['pdf', 'txt', 'blocks', 'uid'])
    pdf_path = Path(__file__).parent / 'test_data' / '001_formatted_pdf.pdf'
    txt_path = Path(__file__).parent / 'test_data' / '001_text.txt'
    block_path = Path(__file__).parent / 'test_data' / '001_blocks.csv'
    uid = '001'
    return(test_data(pdf_path, txt_path, block_path, uid))
