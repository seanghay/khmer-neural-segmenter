from typing import List

__version__: str

class KhmerSegmenter:
  def __init__(self) -> None: ...
  def tokenize(self, text: str) -> List[str]:
    """Segment Khmer text and return a list of words."""
    ...
  def __call__(self, text: str) -> List[str]:
    """Segment Khmer text and return a list of words."""
    ...

def tokenize(text: str) -> List[str]:
  """Segment Khmer text and return a list of words."""
  ...

def normalize(text: str) -> str:
  """Normalize and reorder Khmer character"""
  ...
