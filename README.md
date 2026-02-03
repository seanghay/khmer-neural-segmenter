# Khmer Neural Segmenter

A fast Khmer word segmentation library.

## Installation

```
pip install kns
```

## Usage

```python
from kns import tokenize, segment

# Returns a list of words
words = tokenize("សួស្តីបងប្អូន")
# ['សួស្តី', 'បង', 'ប្អូន']

# Returns a pipe-delimited string
text = segment("សួស្តីបងប្អូន")
# 'សួស្តី|បង|ប្អូន'
```

You can also use the class-based API if you prefer:

```python
from kns import KhmerSegmenter

segmenter = KhmerSegmenter()
words = segmenter.tokenize("សួស្តីបងប្អូន")
# or
words = segmenter("សួស្តីបងប្អូន")
```

## License

MIT