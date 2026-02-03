# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Khmer neural word segmentation using BiGRU + CRF architecture. Combines deep learning with linguistic normalization for Khmer (Cambodian) text processing.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Lint code
ruff check .

# Run model test (basic forward/backward pass)
python model.py

# Generate training data (requires external text source)
python generate.py
```

## Architecture

### Neural Model (`model.py`)
BiGRU + CRF sequence tagging model:
- Embedding layer → Bidirectional GRU → Linear → CRF
- Training: returns negative log-likelihood loss
- Inference: returns decoded tag sequences via CRF

### Data Pipeline (`generate.py`)
Multi-stage segmentation pipeline:
1. Regex pre-segmentation (separates Khmer script, numbers, other)
2. Deep segmentation using pre-trained `khmersegment` model
3. Khmer normalization via `khnormal()`
4. Non-Khmer tokenization via NLTK TweetTokenizer

### Khmer Normalization (`khnormal.py`)
Unicode normalization for Khmer text (U+1780 to U+17DD):
- 13 character categories (Base, Robat, Coeng, Shift, vowels, modifiers)
- Supports Modern Khmer ("km") and Middle Khmer ("xhm") variants
- Syllable reordering based on character categories

## Code Style

- 2-space indentation (configured in `ruff.toml`)

## Key Dependencies

- `pytorch-crf`: CRF layer for sequence tagging
- `khmersegment`: Pre-trained Khmer segmentation (model in `assets/`)
- `regex`: Extended regex for Unicode handling


### Khmer Character Cluster Regex

```python
"[\u1780-\u17FF](\u17D2[\u1780-\u17FF]|[\u17B6-\u17D1\u17D3\u17DD])*"
```

### Tags

- `B-WORD`
- `I-WORD`
- `0`

### Data format

```
តំបន់
វាល
ទំនាប
ចាប់
ពី
ថ្ងៃ
ទី
៥
 
ដល់
ថ្ងៃ
ទី
៧
 
ខែ
វិច្ឆិកា
 
អាច
មាន
ភ្លៀង
ធ្លាក់
ជាមួយ
ផ្គរ
រន្ទះ
 
និង
ខ្យល់
កន្ត្រាក់
គ្រប
 
ដណ្តប់
លើ
ផ្ទៃ
ដី
៣០
%
ភ្នំ
ពេញ
 
៖
 
ក្រសួង
ធនធាន
ទឹក
 
និង
ឧតុនិយម
 
បាន
ចេញ
ព្រឹត្តិ
បត្រ
ព័ត៌មាន
ស្ដី
ពី
ស្ថានភាព
ធាតុ
អាកាស
នៅ
ព្រះ
រាជាណាចក្រ
កម្ពុជា
 
សម្រាប់
ថ្ងៃ
ទី
៥
 
ខែ
វិច្ឆិកា
 
ឆ្នាំ
២០២៥
 
៖ព្យុះ
ទី
២៥
ឈ្មោះ
 
កាល
ម៉េហ្គឹ
(
Kalmaegi
)
```