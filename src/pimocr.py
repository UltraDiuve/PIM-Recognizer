"""OCR wrapper module

This module defines various classes to run OCR functionnalities from several
libraries or online tools.
"""
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as mpatch

import pyocr


class BaseOCR(object):
    """Abstract base class for OCR modules

    This class defines the basic model of OCR subclasses and should not be
    instanciated.
    """
    def __init__(self, wrapper=None, tool_name=None):
        self.wrapper = wrapper
        self.tool_name = tool_name

    def __str__(self):
        print(':'.join([self.wrapper, self.tool_name]))

    def set_file(self, path=None, filename=None):
        full_path = os.path.join(path, filename)
        self.image = Image.open(full_path)
        self.mp_image = mpimg.imread(full_path)
        self.result = None

    def show(self, ax=None):
        if self.result is None:
            raise RuntimeError('Tool has not been run prior to showing')
        ax.imshow(self.mp_image)
        ax.tick_param(bottom=False, left=False)
        ax.set_xlabel('Word count: ' + str(self.count_words()))

    def get_result(self, *args, **kwargs):
        return(self.result)

    def run_tool(self):
        if self.file is None:
            raise RuntimeError('File has not been set prior to running tool')

    def count_words(self):
        if self.result is None:
            raise RuntimeError('Tool has not been run prior to '
                               'counting results')


class PyocrWrappedOCR(BaseOCR):
    """Abstract class for pyocr wrapped tools

    This class defines some functions for OCR tools based on pyocr library, and
    should not be instanciated.
    """
    def __init__(self, tool_name=None, builder=None, **kwargs):
        """Constructor method for pyocr wrapped tools.

        tool_name arg should always be provided, and from the following list
        (see pyocr documentation): 'Tesseract (sh)', 'Tesseract (C-API)' or
        'Cuneiform (sh)'.
        """
        if tool_name is None:
            raise RuntimeError('No tool as been specified')
        tool_list = map(lambda x: x.get_name(), pyocr.get_available_tools())
        if tool_name not in tool_list:
            raise RuntimeError(f'Tool {tool_name} not installed on current '
                               'env.')
        for pyocr_tool in pyocr.get_available_tools():
            if pyocr_tool.get_name() == tool_name:
                self.tool = pyocr_tool
                break

        builder_args = ['tesseract_layout']
        filtered_kwargs = {key: val for key, val in kwargs.items()
                           if key in builder_args}
        self.builder = builder(**filtered_kwargs)
        super().__init__(tool_name=tool_name, wrapper='pyocr')

    def run_tool(self, lang='fra', **kwargs):
        self.result = self.tool.image_to_string(
            self.image,
            lang=lang,
            builder=self.builder
        )

    def get_result(self, *args, **kwargs):
        pass


class TextOCR(BaseOCR):
    """Abstract class for text only OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_words(self):
        super().count_words()
        words = self.result.split()
        return(len(words))


class WordBoxOCR(BaseOCR):
    """Abstract class for WordBox OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_words(self):
        super().count_words()
        return(len(self.result))

    def show(self, ax=None, **kwargs):
        super().show(ax=ax, **kwargs)
        ax.set_title('TODO ! Montre quon a bien maj ou il fo!')


class PyocrTextOCR(PyocrWrappedOCR, TextOCR):
    """Class that instantiate a text only pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.TextBuilder, **kwargs)


class PyocrWordBoxOCR(PyocrWrappedOCR, WordBoxOCR):
    """Class that instantiate a wordbox pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.WordBoxBuilder, **kwargs)


class WordBox(object):
    """Represents a wordbox object returned by an OCR tool

    This class instanciate a wordbox object that can be retrieved from an OCR
    tool and then be drawn on an axes.
    """
    def __init__(self, box_position):
        self.box_position = box_position
        self.x = box_position[0][0]
        self.y = box_position[0][1]
        self.width = box_position[1][0] - self.x
        self.height = box_position[1][1] - self.y

    def to_rect_coord(self):
        return(((self.x, self.y), self.width, self.height))

    def draw(self, ax=None, fill=False, color='red', lw=2, **kwargs):
        ax.add_patch(mpatch.Rectangle(*self.to_rect_coord(),
                                      fill=fill,
                                      color=color,
                                      lw=lw))
