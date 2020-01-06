"""OCR wrapper module

This module defines various classes to run OCR functionnalities from several
libraries or online tools.
"""
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

    def set_file(self):
        pass

    def show(self, ax):
        pass

    def get_results(self, *args, **kwargs):
        pass

    def count_results(self):
        pass


class PyorcWrappedOCR(BaseOCR):
    """Abstract class for pyocr wrapped tools

    This class defines some functions for OCR tools based on pyocr library, and
    should not be instanciated.
    """
    def __init__(self, tool_name=None, **kwargs):
        """Constructor method for pyocr wrapped tools.

        tool_name arg should always be provided, and from the following list
        (see pyocr documentation): 'Tesseract (sh)', 'Tesseract (C-API)' or
        'Cuneiform (sh)'.
        """
        if tool_name is None:
            raise ValueError("Please specify a tool.")
        tool_list = map(lambda x: x.get_name(), pyocr.get_available_tools())
        if tool_name not in tool_list:
            raise ValueError(f"Tool {tool_name} not installed on current env.")
        super().__init__(tool_name=tool_name, wrapper='pyocr')

    def set_file(self):
        pass

    def show(self, ax):
        pass

    def get_results(self, *args, **kwargs):
        pass

    def count_results(self):
        pass


class TextOCR(BaseOCR):
    """Abstract class for text only OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self):
        super.__init__(**kwargs)

    def set_file(self):
        pass

    def show(self, ax):
        pass

    def get_results(self, *args, **kwargs):
        pass

    def count_results(self):
        pass


class PyocrTextOCR(PyorcWrappedOCR, TextOCR):
    """Class that instantiate a text only pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        self.builder = pyocr.builders.TextBuilder()
        super().__init__(**kwargs)

    def set_file(self):
        pass

    def show(self, ax):
        pass

    def get_results(self, *args, **kwargs):
        pass

    def count_results(self):
        pass
