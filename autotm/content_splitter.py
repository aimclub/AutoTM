from abc import ABC


class BaseTextSplitter(ABC):

    def text_process(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def text_split(self):
        raise NotImplementedError

class TextSplitter(BaseTextSplitter):
    def content_splitting(self):
        raise NotImplementedError

