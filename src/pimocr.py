"""OCR wrapper module

This module defines various classes to run OCR functionnalities from several
libraries or online tools.
"""
import pyocr
from PIL import Image
import os


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

    def set_file(self, path, filename):
        self.image = Image.open(os.path.join(path, filename))

    def show(self, ax):
        pass

    def get_result(self, *args, **kwargs):
        return(self.result)

    def run_tool(self):
        if self.file is None:
            raise RuntimeError("File has not been set prior to running tool")

    def count_result(self):
        pass


class PyocrWrappedOCR(BaseOCR):
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
        for pyocr_tool in pyocr.get_available_tools():
            if pyocr_tool.get_name() == tool_name:
                self.tool = pyocr_tool
                break
        super().__init__(tool_name=tool_name, wrapper='pyocr')

    def show(self, ax):
        pass

    def run_tool(self, lang='fra', **kwargs):
        self.result = self.tool.image_to_string(
            self.image,
            lang=lang,
            builder=self.builder
        )

    def get_result(self, *args, **kwargs):
        pass

    def count_result(self):
        words = self.result.split()
        return(len(words))


class TextOCR(BaseOCR):
    """Abstract class for text only OCR functionnalities

    This class describes the text only OCR functionnalities, and
    should not be instanciated.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_file(self):
        pass

    def show(self, ax):
        pass

    def get_result(self, *args, **kwargs):
        pass

    def count_result(self):
        pass


class PyocrTextOCR(PyocrWrappedOCR, TextOCR):
    """Class that instantiate a text only pyocr wrapped tool

    This class instanciates the pyocr raw text functionnality.
    """
    def __init__(self, **kwargs):
        self.builder = pyocr.builders.TextBuilder()
        super().__init__(**kwargs)

    def set_file(self, path, filename):
        super().set_file(path, filename)

    def show(self, ax):
        pass

    def get_result(self, *args, **kwargs):
        pass

    def count_result(self):
        pass
