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
        ax.tick_params(which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        ax.set_xlabel('Word count: ' + str(self.count_words()))

    def run_tool(self):
        if self.image is None:
            raise RuntimeError('File has not been set prior to running tool')

    def count_words(self):
        if self.result is None:
            raise RuntimeError('Tool has not been run prior to '
                               'counting results')

    def structure_results(self):
        pass


class PyocrWrappedOCR(BaseOCR):
    """Abstract class for pyocr wrapped tools

    This class defines some functions for OCR tools based on pyocr library, and
    should not be instanciated.
    """
    def __init__(self, tool_name=None, builder=None,
                 tesseract_layout=None, **kwargs):
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
        filtered_kwargs = {key: val for key, val in locals().items()
                           if key in builder_args and val is not None}
        self.builder = builder(**filtered_kwargs)
        super().__init__(tool_name=tool_name, wrapper='pyocr', **kwargs)

    def run_tool(self, lang='fra', **kwargs):
        self.result = self.tool.image_to_string(
            self.image,
            lang=lang,
            builder=self.builder
        )
        super().run_tool(**kwargs)


class TextOCR(BaseOCR):
    """Abstract class for text only OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_words(self):
        super().count_words()
        return(len(self.words))


class WordBoxOCR(BaseOCR):
    """Abstract class for WordBox OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_words(self):
        super().count_words()
        return(len(self.words))

    def show(self, ax=None, fill=True, color='red', lw=2, annotate=True,
             **kwargs):
        super().show(ax=ax, **kwargs)
        for wordbox in self.wordboxes:
            wordbox.draw(ax=ax, fill=fill, color=color, lw=lw, **kwargs)
            if annotate:
                wordbox.annotate(ax, where='center', **kwargs)

    def structure_results(self, **kwargs):
        self.words = [wordbox.content for wordbox in self.wordboxes]
        super().structure_results(**kwargs)


class PyocrTextOCR(PyocrWrappedOCR, TextOCR):
    """Class that instantiates a text only pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.TextBuilder, **kwargs)

    def run_tool(self, **kwargs):
        super().run_tool(**kwargs)
        self.structure_results(**kwargs)

    def structure_results(self, **kwargs):
        self.words = self.result.split()
        super().structure_results()


class PyocrWordBoxOCR(PyocrWrappedOCR, WordBoxOCR):
    """Class that instantiates a wordbox pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.WordBoxBuilder, **kwargs)

    def run_tool(self, **kwargs):
        super().run_tool(**kwargs)
        self.structure_results(**kwargs)

    def structure_results(self, **kwargs):
        self.wordboxes = [PyocrWordBox(pyocrbox) for pyocrbox in self.result]
        super().structure_results(**kwargs)


class WordBox(object):
    """Represents a generic wordbox object returned by an OCR tool

    This class instanciates a wordbox object that can be retrieved from an OCR
    tool and then be drawn on an axes.
    """
    def __init__(self, x, y, width, height, content):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.content = content

    def to_rect_coord(self):
        return(((self.x, self.y), self.width, self.height))

    def draw(self, ax=None, fill=False, color='red', lw=2, **kwargs):
        ax.add_patch(mpatch.Rectangle(*self.to_rect_coord(),
                                      fill=fill,
                                      color=color,
                                      lw=lw,
                                      **kwargs))

    def annotate(self, ax, where='above left', color='blue', **kwargs):

        if where == 'above left':
            xy = (self.x, self.y)
            verticalalignment = 'baseline'
            horizontalalignment = 'left'
        if where == 'center':
            xy = self.center()
            verticalalignment = 'center'
            horizontalalignment = 'center'
        ax.annotate(self.content, xy, verticalalignment=verticalalignment,
                    horizontalalignement=horizontalalignment, color=color,
                    **kwargs)

    def center(self):
        """Returns the center of the box
        """
        return((self.x + self.width / 2, self.y + self.height / 2))


class PyocrWordBox(WordBox):
    """Represents a wordbox object returned by pyocr

    This class instanciates a wordbox object that can be retrieved from pyocr,
    be it through WordBox builder or LineBox builder (wordboxes are children
    of lineboxes).
    """
    def __init__(self, pyocrbox):
        self.pyocrbox = pyocrbox
        x = pyocrbox.position[0][0]
        y = pyocrbox.position[0][1]
        width = pyocrbox.position[1][0] - x
        height = pyocrbox.position[1][1] - y
        content = pyocrbox.content
        super().__init__(x=x, y=y, width=width, height=height, content=content)

    def confidence(self):
        return(self.pyocrbox.confidence)

    def is_empty(self):
        return(self.pyocrbox.content.strip() == '')


class FilterableOCR(BaseOCR):
    """Abstract class for tools whose results can be filtered (e.g. confidence)

    This class describes how tools whose results can be filtered should
    behave. It applies to tools that provide confidence parameters, but also
    to the ones that can return objects (lines, boxes, ...) with empty content.
    """
    def __init__(self, conf_low=0, conf_high=100, filter_empty=True,
                 filter_conf=True, **kwargs):
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.filter_empty = filter_empty
        self.filter_conf = filter_conf
        super().__init__(**kwargs)
        print('TODO !!! Where to filter ?')
