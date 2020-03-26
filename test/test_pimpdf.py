"""
Unit test for pimpdf module
"""

import pytest
from pathlib import Path
from collections import namedtuple
from io import BytesIO
import pandas as pd
import re

from src.pimpdf import PDFDecoder


@pytest.fixture
def test_data_002():
    test_data = namedtuple('Test_data', ['pdf', 'txt', 'blocks', 'uid'])
    pdf_path = Path(__file__).parent / 'test_data' / '002_formatted_pdf.pdf'
    txt_path = Path(__file__).parent / 'test_data' / '002_text.txt'
    block_path = Path(__file__).parent / 'test_data' / '002_blocks.csv'
    uid = '002'
    return(test_data(pdf_path, txt_path, block_path, uid))


@pytest.fixture
def test_data_no_file():
    test_data = namedtuple('Test_data', ['pdf', 'txt', 'blocks', 'uid'])
    pdf_path = Path(__file__).parent / 'test_data' / 'NO_SUCH_FILE.pdf'
    txt_path = Path(__file__).parent / 'test_data' / 'NO_SUCH_FILE.txt'
    block_path = Path(__file__).parent / 'test_data' / 'NO_SUCH_FILE.csv'
    uid = '003'
    return(test_data(pdf_path, txt_path, block_path, uid))


@pytest.fixture
def test_path_series(test_data_001, test_data_002):
    idx = pd.Index([test_data_001.uid, test_data_002.uid], name='uid')
    data = [test_data_001.pdf, test_data_002.pdf]
    return(pd.DataFrame(data, index=idx, columns=['path']))


@pytest.fixture
def make_target_series():
    def transform_target(test_data):
        ds = pd.read_csv(test_data.blocks,
                         encoding='utf-8-sig',
                         sep=';',
                         header=None,
                         names=['blocks'],
                         squeeze=True,
                         )
        idx = pd.DataFrame([test_data.uid] * len(ds), columns=['idx'])
        ds = pd.concat([ds, idx], axis=1).set_index('idx')['blocks']
        form_feed_ds = pd.Series(['\x0c'], name='blocks',
                                 index=[test_data.uid])
        ds = pd.concat([ds, form_feed_ds], axis=0)
        return(ds)
    return(transform_target)


@pytest.fixture
def splitter_func():
    def splitter(x):
        return(re.split(r'\n *\n', x))
    return(splitter)


class TestPDFDecoder(object):

    def test_path_to_text(self, test_data_001):
        target = Path(test_data_001.txt).read_text()
        target += '\x0c'
        assert PDFDecoder.path_to_text(test_data_001.pdf) == target

    def test_path_to_text_2(self, test_data_002):
        target = Path(test_data_002.txt).read_text(encoding='utf-8-sig')
        target = target + '\x0c'
        assert PDFDecoder.path_to_text(test_data_002.pdf) == target

    def test_path_to_text_no_file(self):
        assert PDFDecoder.path_to_text('incorect/path',
                                       missing_file='ignore') == ''
        with pytest.raises(FileNotFoundError):
            PDFDecoder.path_to_text('incorect/path')

    def test_path_to_text_corrupted(self):
        path = Path(__file__).parent / 'test_data' / '003_corrupted_pdf.pdf'
        assert PDFDecoder.path_to_text(path, missing_file='raise') == ''

    def test_path_to_text_incorrect_params(self):
        with pytest.raises(ValueError):
            PDFDecoder.path_to_text('', missing_file='incorrect')

    def test_path_to_blocks(self, test_data_001, splitter_func):
        target = pd.read_csv(test_data_001.blocks,
                             encoding='utf-8-sig',
                             sep=';',
                             header=None,
                             names=['blocks'],
                             squeeze=True,
                             )
        target = list(target)
        target.append('\x0c')
        blocks = PDFDecoder.path_to_blocks(test_data_001.pdf,
                                           split_func=splitter_func)
        assert target == blocks

    def test_path_to_blocks_2(self, test_data_002, splitter_func):
        target = pd.read_csv(test_data_002.blocks,
                             encoding='utf-8-sig',
                             sep=';',
                             header=None,
                             names=['blocks'],
                             squeeze=True,
                             )
        target = list(target)
        target.append('\x0c')
        blocks = PDFDecoder.path_to_blocks(test_data_002.pdf,
                                           split_func=splitter_func)
        assert target == blocks

    def test_path_to_blocks_series(self, test_data_001, test_data_002,
                                   make_target_series, splitter_func):
        target = make_target_series(test_data_001)
        blocks_series = (PDFDecoder
                         .path_to_blocks_series(test_data_001.pdf,
                                                split_func=splitter_func,
                                                index=test_data_001.uid)
                         )
        assert target.equals(blocks_series)
        target = make_target_series(test_data_002)
        blocks_series = (PDFDecoder
                         .path_to_blocks_series(test_data_002.pdf,
                                                split_func=splitter_func,
                                                index=test_data_002.uid)
                         )
        assert target.equals(blocks_series)

    def test_paths_to_blocks(self, test_data_001, test_data_002,
                             make_target_series, splitter_func):
        target_ds = pd.concat([make_target_series(test_data_001),
                               make_target_series(test_data_002)],
                              axis=0)
        input = pd.Series([test_data_001.pdf, test_data_002.pdf],
                          index=[test_data_001.uid, test_data_002.uid])
        output = PDFDecoder.paths_to_blocks(input, split_func=splitter_func)
        assert output.equals(target_ds)

    def test_paths_to_blocks_with_missing(self, test_data_001, test_data_002,
                                          make_target_series, splitter_func,
                                          test_data_no_file):
        no_file_ds = pd.Series([''], index=[test_data_no_file.uid])
        target_ds = pd.concat([make_target_series(test_data_001),
                               make_target_series(test_data_002),
                               no_file_ds],
                              axis=0)
        input = pd.Series([test_data_001.pdf,
                           test_data_002.pdf,
                           test_data_no_file.pdf,
                           ],
                          index=[test_data_001.uid,
                                 test_data_002.uid,
                                 test_data_no_file.uid,
                                 ])
        output = PDFDecoder.paths_to_blocks(input,
                                            split_func=splitter_func,
                                            missing_file='ignore')
        assert output.equals(target_ds)
        output = PDFDecoder.threaded_paths_to_blocks(input,
                                                     split_func=splitter_func,
                                                     missing_file='ignore')
        assert output.equals(target_ds)

        with pytest.raises(FileNotFoundError):
            PDFDecoder.threaded_paths_to_blocks(input,
                                                split_func=splitter_func,
                                                missing_file='raise',
                                                )

    def test_content_to_text_to_blocks(self, test_data_001, test_data_002,
                                       make_target_series, splitter_func,
                                       ):
        target_ds = pd.concat([make_target_series(test_data_001),
                               make_target_series(test_data_002)],
                              axis=0)
        content_list = []
        for test_data in [test_data_001, test_data_002]:
            with open(test_data.pdf, mode='rb') as content:
                content_list.append(content.read())
        content_ds = pd.Series(content_list, index=[test_data_001.uid,
                                                    test_data_002.uid])
        text_ds = PDFDecoder.threaded_contents_to_text(content_ds)
        blocks_ds = (PDFDecoder
                     .threaded_texts_to_blocks(text_ds,
                                               split_func=splitter_func))
        assert blocks_ds.equals(target_ds)

    def test_invalid_return_type(self):
        text_series = pd.Series(['first text\n\n2lines',
                                 'second text\n\nline\n\nlineagain'])
        with pytest.raises(ValueError):
            PDFDecoder.threaded_texts_to_blocks(text_series,
                                                return_type='incorrect_input',
                                                )
        with pytest.raises(ValueError):
            PDFDecoder.text_to_blocks_series(text_series.iloc[0],
                                             return_type='incorrect_input',
                                             )

    def test_texts_to_blocks_as_list(self):
        text_series = pd.Series(['first text\n\n2lines',
                                 'second text\n\nline\n\nlineagain'],
                                index=pd.Index(['001', '002']),
                                )
        blocks_ds = (PDFDecoder
                     .threaded_texts_to_blocks(text_series,
                                               return_type='as_list',
                                               )
                     )
        target = pd.Series([['first text', '2lines'],
                            ['second text', 'line', 'lineagain']],
                           index=pd.Index(['001', '002']))
        assert blocks_ds.equals(target)
        blocks_ds = PDFDecoder.text_to_blocks_series('first text\n\n2lines',
                                                     index='001',
                                                     return_type='as_list')
        print(blocks_ds)
        print(target.loc['001'])
        assert blocks_ds.loc['001'] == target.loc['001']

    def test_empty_content_raise(self,
                                 test_data_001,
                                 test_data_002,
                                 make_target_series,
                                 splitter_func,
                                 ):
        content_list = []
        for test_data in [test_data_001, test_data_002]:
            with open(test_data.pdf, mode='rb') as content:
                content_list.append(content.read())
        content_list[0] = None
        content_ds = pd.Series(content_list, index=[test_data_001.uid,
                                                    test_data_002.uid])
        # Case where error should be raised (content input is None)
        with pytest.raises(RuntimeError):
            PDFDecoder.threaded_contents_to_text(content_ds,
                                                 none_content='raise')

    def test_empty_content_none(self,
                                test_data_001,
                                test_data_002,
                                make_target_series,
                                splitter_func,
                                ):
        content_list = []
        for test_data in [test_data_001, test_data_002]:
            with open(test_data.pdf, mode='rb') as content:
                content_list.append(content.read())
        content_list[0] = None
        content_ds = pd.Series(content_list, index=[test_data_001.uid,
                                                    test_data_002.uid])
        # Passing case where content input is None
        texts_ds = (PDFDecoder
                    .threaded_contents_to_text(content_ds,
                                               none_content='to_empty'))
        assert texts_ds.iloc[0] == ''
        assert (texts_ds.iloc[1] ==
                (Path(test_data_002.txt).read_text(encoding='utf-8-sig')
                 + '\x0c'))

    def test_empty_content_empty(self,
                                 test_data_001,
                                 test_data_002,
                                 make_target_series,
                                 splitter_func,
                                 ):
        # Passing case where content input is empty string
        content_list = []
        for test_data in [test_data_001, test_data_002]:
            with open(test_data.pdf, mode='rb') as content:
                content_list.append(content.read())
        content_list[0] = b''
        content_ds = pd.Series(content_list, index=[test_data_001.uid,
                                                    test_data_002.uid])
        texts_ds = (PDFDecoder
                    .threaded_contents_to_text(content_ds,
                                               none_content='to_empty'))
        assert texts_ds.iloc[0] == ''
        assert (texts_ds.iloc[1] ==
                (Path(test_data_002.txt).read_text(encoding='utf-8-sig')
                 + '\x0c'))

    def test_invalid_arg(self,
                         test_data_001,
                         test_data_002,
                         make_target_series,
                         splitter_func,
                         ):
        content_list = []
        for test_data in [test_data_001, test_data_002]:
            with open(test_data.pdf, mode='rb') as content:
                content_list.append(content.read())
        content_ds = pd.Series(content_list, index=[test_data_001.uid,
                                                    test_data_002.uid])
        # Unexpected none_content argument
        with pytest.raises(ValueError):
            (PDFDecoder.
             threaded_contents_to_text(content_ds,
                                       none_content='incorrect_input'))

    def test_invalid_none_content_arg(self):
        with pytest.raises(ValueError):
            PDFDecoder.content_to_text(b'', none_content='incorrect_input')

    def test_none_content(self):
        content = BytesIO(b'')
        with pytest.raises(RuntimeError):
            PDFDecoder.content_to_text(content, none_content='raise')
        assert (PDFDecoder.content_to_text(content, none_content='to_empty')
                == '')

    def test_incorrect_content_type(self):
        with pytest.raises(AttributeError):
            PDFDecoder.content_to_text(b'', none_content='to_empty')
