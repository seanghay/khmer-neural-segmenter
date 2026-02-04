# Khmer Neural Segmenter

A fast Khmer word segmentation library.

<img src="assets/graph.png" alt="" width=500>

## Installation

```
pip install khmerns
```

## Usage

```python
from khmerns import tokenize

# Returns a list of words
words = tokenize("សួស្តីបងប្អូន")
# ['សួស្តី', 'បង', 'ប្អូន']
```

You can also use the class-based API if you prefer:

```python
from khmerns import KhmerSegmenter

segmenter = KhmerSegmenter()
words = segmenter.tokenize("សួស្តីបងប្អូន")
# or
words = segmenter("សួស្តីបងប្អូន")
```

## License

MIT