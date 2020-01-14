"""OCR wrapper module

This module defines various classes to run OCR functionnalities from several
libraries or online tools.

Key concepts:
- run_tool: will run the tool and get its return into the 'result' atttribute.
The return of the tool is kept raw.
- parse_result: will use the content of the 'result' attribute to provide the
instance with its type-related attributes (internals). e.g. the lines for a
LineBoxOCR.
- refresh_internals: will (re)syncrhonize the internals of super levels with
the content of the current instance. e.g. will (re)compute the WordBoxes for a
LineBoxOCR.
"""
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as mpatch
import requests

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

    def show(self, ax):
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

    def refresh_internals(self):
        pass

    def parse_result(self):
        self.refresh_internals()


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

    def filter(self, conf_level=80):
        self.reset_filter()
        self.raw_wordboxes = self.wordboxes.copy()
        self.wordboxes = []
        if self.filter_conf:
            for wordbox in self.raw_wordboxes:
                if wordbox.confidence() >= conf_level:
                    self.wordboxes.append(wordbox)
        self.refresh_internals()

    def reset_filter(self):
        try:
            self.wordboxes = self.raw_wordboxes.copy()
            del(self.raw_wordboxes)
        except AttributeError:
            pass
        self.refresh_internals()


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


class AzureWrappedOCR(BaseOCR):
    """Abstract class for Microsoft Azure wrapped OCR tools

    This class defines some functions for OCR tools based on Microsoft Azure
    services.
    """
    def __init__(self, endpoint, suffix='/vision/v2.0/ocr', **kwargs):
        self.url = endpoint + suffix
        super().__init__(wrapper='Azure', tool_name='OCR', **kwargs)

    def set_file(self, path=None, filename=None, **kwargs):
        full_path = os.path.join(path, filename)
        self.binaryfile = open(full_path, 'rb')
        super().set_file(path=path, filename=filename, **kwargs)

    def run_tool(self, subscriptionkey, proxies=None, **kwargs):
        super().run_tool(**kwargs)
        headers = {'Content-Type': 'application/octet-stream',
                   'Ocp-Apim-Subscription-Key': subscriptionkey}
        self.result = requests.post(self.url,
                                    proxies=proxies,
                                    data=self.binaryfile,
                                    headers=headers).json()


class TextOCR(BaseOCR):
    """Abstract class for text only OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def count_words(self):
        super().count_words()
        return(len(self.words))


class WordBoxOCR(TextOCR):
    """Abstract class for wordbox OCR functionnalities

    This class describes the OCR functionnalities regarding wordboxes, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_words(self):
        super().count_words()
        return(len(self.words))

    def show(self, ax, what, annotate=True, format_box=None,
             format_annotate=None, **kwargs):
        super().show(ax, what, **kwargs)
        if what == 'words' or 'words' in what:
            for wordbox in self.wordboxes:
                wordbox.show(ax,
                            format_box=format_box,
                            format_annotate=format_annotate,
                            **kwargs)

    def refresh_internals(self, **kwargs):
        self.words = [wordbox.content for wordbox in self.wordboxes]
        super().refresh_internals(**kwargs)


class LineBoxOCR(WordBoxOCR):
    """Abstract class for linebox OCR functionnalities

    This class describes the OCR functionnalities regarding lineboxes, and
    should not be instanciated.
    """
    def refresh_internals(self, **kwargs):
        self.wordboxes = [wordbox
                          for linebox in self.lineboxes
                          for wordbox in linebox.childrenboxes]
        super().refresh_internals(**kwargs)

    def show(self, ax, what, annotate=True, format_line=None,
             format_annotate_line=None, **kwargs):
        super().show(ax=ax, **kwargs)
        if what == 'lines' or 'lines' in what:
            for linebox in self.lineboxes:
                linebox.show(ax,
                            format_box=format_line,
                            format_annotate=format_annotate_line,
                            **kwargs)


class PyocrTextOCR(PyocrWrappedOCR, TextOCR):
    """Class that instantiates a text only pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.TextBuilder, **kwargs)

    def run_tool(self, **kwargs):
        super().run_tool(**kwargs)
        self.parse_result(**kwargs)

    def parse_result(self, **kwargs):
        self.words = self.result.split()
        super().parse_result(**kwargs)


class PyocrWordBoxOCR(PyocrWrappedOCR, WordBoxOCR, FilterableOCR):
    """Class that instantiates a wordbox pyocr wrapped tool

    This class instanciates the pyocr wordboxes recognition functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.WordBoxBuilder, **kwargs)

    def run_tool(self, **kwargs):
        super().run_tool(**kwargs)
        self.parse_result(**kwargs)

    def parse_result(self, **kwargs):
        self.wordboxes = [PyocrWordBox(pyocrwordbox)
                          for pyocrwordbox in self.result]
        super().parse_result(**kwargs)


class PyocrLineBoxOCR(PyocrWrappedOCR, LineBoxOCR):
    """Class that instantiates a linebox pyocr wrapped tool

    This class instanciates the pyocr lineboxes recognition functionnality.
    """
    def __init__(self, **kwargs):
        super().__init__(builder=pyocr.builders.LineBoxBuilder, **kwargs)

    def run_tool(self, **kwargs):
        super().run_tool(**kwargs)
        self.parse_result(**kwargs)

    def parse_result(self, **kwargs):
        self.lineboxes = [PyocrLineBox(pyocrlinebox)
                          for pyocrlinebox in self.result]
        super().parse_result(**kwargs)


class Box(object):
    """Represents a generic box object returned by an OCR tool

    This class instanciates a box object that can be retrieved from an OCR
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

    def draw(self, ax, **kwargs):
        ax.add_patch(mpatch.Rectangle(*self.to_rect_coord(), **kwargs))

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
                    horizontalalignment=horizontalalignment, color=color,
                    **kwargs)

    def center(self):
        """Returns the center of the box
        """
        return((self.x + self.width / 2, self.y + self.height / 2))

    def show(self, ax, annotate=True, format_box=None, format_annotate=None):
        default_box_format = {
            'color': 'red',
            'lw': 2,
            'fill': True
        }
        if format_box is not None:
            default_box_format.update(format_box)
        where = 'center' if default_box_format['fill'] else 'above left'
        default_annotate_format = {
            'color': 'blue',
            'where': where
        }
        if format_annotate is not None:
            default_annotate_format.update(format_annotate)
        self.draw(ax=ax, **default_box_format)
        if annotate:
            self.annotate(ax, **default_annotate_format)

    def is_empty(self):
        return(self.content.strip() == '')


class PyocrWordBox(Box):
    """Represents a wordbox object returned by pyocr

    This class instanciates a wordbox object that can be retrieved from pyocr,
    be it through wordbox builder or linebox builder (wordboxes are children
    of lineboxes).
    """
    def __init__(self, pyocrwordbox):
        self.pyocrwordbox = pyocrwordbox
        x = pyocrwordbox.position[0][0]
        y = pyocrwordbox.position[0][1]
        width = pyocrwordbox.position[1][0] - x
        height = pyocrwordbox.position[1][1] - y
        content = pyocrwordbox.content
        super().__init__(x=x, y=y, width=width, height=height, content=content)

    def confidence(self):
        return(self.pyocrwordbox.confidence)


class PyocrLineBox(Box):
    """Represents a linebox object returned by pyocr

    This class instanciates a linebox object that can be retrieved from pyocr,
    through the linebox builder.
    """
    def __init__(self, pyocrlinebox):
        self.pyocrlinebox = pyocrlinebox
        x = pyocrlinebox.position[0][0]
        y = pyocrlinebox.position[0][1]
        width = pyocrlinebox.position[1][0] - x
        height = pyocrlinebox.position[1][1] - y
        content = pyocrlinebox.content
        self.childrenboxes = [PyocrWordBox(pyocrwordbox)
                              for pyocrwordbox in pyocrlinebox.word_boxes]
        super().__init__(x=x, y=y, width=width, height=height, content=content)

    def confidence(self):
        raise NotImplementedError('Pyocr lineboxes do not have confidence')
