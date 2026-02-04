# Khmer Neural Segmenter

A fast Khmer word segmentation library.

<img src="img/graph.png" alt="" width=500>

## Installation

```
pip install khmerns
```

## Usage

```python
from khmerns import tokenize, normalize

# Returns a list of words
words = tokenize("សួស្តីបងប្អូន")
# => ['សួស្តី', 'បង', 'ប្អូន']

# normalize and reorder Khmer characters
words = tokenize(normalize("សួស្តីបងប្អូន"))
# => ['សួស្តី', 'បង', 'ប្អូន']
```

You can also use the class-based API if you prefer:

```python
from khmerns import KhmerSegmenter

segmenter = KhmerSegmenter()

words = segmenter.tokenize("សួស្តីបងប្អូន")
# or

words = segmenter("សួស្តីបងប្អូន")
```

## Training

The training pipeline lives in the `training/` directory. It trains a BiGRU + CRF model on character-level BIO tags, then converts the result to GGUF for the C++ inference backend.

### Data format

Training data is a plain text file at `training/data/train.txt`. One word per line. Words that appear on consecutive lines are treated as part of the same sentence. The model learns word boundaries from this.

Example `training/data/train.txt`:

```
សួស្តី
បង
ប្អូន
ខ្ញុំ
ទៅ
ផ្សារ
```

Non-Khmer tokens (spaces, punctuation, numbers, Latin text) are tagged as `NON-KHMER`. Khmer tokens get `B-WORD` on the first character and `I-WORD` on the rest.

### Steps

```bash
cd training
pip install -r requirements.txt
```

**1. Prepare training data**

Place your segmented text in `data/train.txt` (one word per line). If you have raw unsegmented Khmer text, you can use the generation script to pre-segment it:

```bash
python generate.py
```

This requires `khmersegment` and a source text file. Edit the path in `generate.py` to point to your raw text.

**2. Train**

```bash
python train.py
```

Trains for 20 epochs with AdamW (lr=1e-5) and ReduceLROnPlateau. Saves `best_model.pt` (best eval loss) and `model.pt` (final). Uses CUDA if available.

**3. Convert to GGUF**

```bash
python convert_to_gguf.py best_model.pt model.gguf
```

This produces a GGUF file (~3.3MB) containing all model weights.

**4. Embed in the C++ binary**

To use the new model in the library, convert the GGUF file to a C header and replace `src/model_data.h`, then rebuild:

```bash
xxd -i model.gguf > ../src/model_data.h
pip install -e ..
```

## License

MIT