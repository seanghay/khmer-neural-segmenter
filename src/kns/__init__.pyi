from typing import List

__version__: str

class KhmerSegmenter:
    def __init__(self) -> None: ...
    def segment(self, text: str) -> str:
        """Segment Khmer text and return pipe-delimited words."""
        ...
    def tokenize(self, text: str) -> List[str]:
        """Segment Khmer text and return a list of words."""
        ...
    def __call__(self, text: str) -> List[str]:
        """Segment Khmer text and return a list of words."""
        ...

def segment(text: str) -> str:
    """Segment Khmer text and return pipe-delimited words."""
    ...

def tokenize(text: str) -> List[str]:
    """Segment Khmer text and return a list of words."""
    ...
