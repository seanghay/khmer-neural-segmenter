#include "tokenizer.h"

// Static member definitions
const int32_t khmer_tokenizer::PAD_ID;
const int32_t khmer_tokenizer::BOS_ID;
const int32_t khmer_tokenizer::EOS_ID;
const int32_t khmer_tokenizer::UNK_ID;

static const char * VOCAB[] = {
  " ", "!", "#", "$", "%", "&", "(", ")", "+", ",",
  "-", ".", "/", "0", "1", "2", "3", "4", "5", "6",
  "7", "8", "9", ":", ";", "=", "?", "@", "A", "B",
  "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
  "W", "X", "Y", "Z", "_", "a", "b", "c", "d", "e",
  "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
  "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
  "z", "«", "°", "»", "á", "é", "ë", "ó", "ö", "ü",
  // Khmer consonants
  "ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ",
  "ដ", "ឋ", "ឌ", "ឍ", "ណ", "ត", "ថ", "ទ", "ធ", "ន",
  "ប", "ផ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ", "ស",
  "ហ", "ឡ", "អ",
  // Khmer independent vowels
  "ឤ", "ឥ", "ឦ", "ឧ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ", "ឯ",
  "ឱ", "ឲ",
  // Khmer dependent vowels
  "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ",
  "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ",
  // Khmer signs
  "ំ", "ះ", "ៈ", "៉", "៊", "់", "៌", "៍", "៏", "័",
  "្",
  // Khmer punctuation
  "។", "៕", "៖", "ៗ", "៘", "៛",
  // Khmer digits
  "០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩",
};

static const size_t VOCAB_SIZE = sizeof(VOCAB) / sizeof(VOCAB[0]);

// Get UTF-8 codepoint length from first byte
static int utf8_len(unsigned char c) {
  if ((c & 0x80) == 0) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

// Decode a UTF-8 sequence to a Unicode codepoint, return bytes consumed
static int utf8_to_codepoint(const unsigned char * p, uint32_t & cp) {
  if ((p[0] & 0x80) == 0) {
    cp = p[0];
    return 1;
  }
  if ((p[0] & 0xE0) == 0xC0) {
    cp = ((uint32_t)(p[0] & 0x1F) << 6) | (p[1] & 0x3F);
    return 2;
  }
  if ((p[0] & 0xF0) == 0xE0) {
    cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    return 3;
  }
  if ((p[0] & 0xF8) == 0xF0) {
    cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12)
       | ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    return 4;
  }
  cp = 0xFFFD; // replacement character
  return 1;
}

void khmer_tokenizer_init(khmer_tokenizer & tok) {
  tok.vocab.clear();
  tok.codepoint_to_id.clear();

  for (size_t i = 0; i < VOCAB_SIZE; i++) {
    tok.vocab.push_back(VOCAB[i]);
    uint32_t cp;
    utf8_to_codepoint(reinterpret_cast<const unsigned char *>(VOCAB[i]), cp);
    tok.codepoint_to_id[cp] = static_cast<int32_t>(i + 4);
  }
}

std::vector<int32_t> khmer_tokenizer_encode(
  const khmer_tokenizer & tok,
  const char * text
) {
  std::vector<int32_t> ids;
  const unsigned char * p = reinterpret_cast<const unsigned char *>(text);

  while (*p) {
    uint32_t cp;
    int len = utf8_to_codepoint(p, cp);

    auto it = tok.codepoint_to_id.find(cp);
    if (it != tok.codepoint_to_id.end()) {
      ids.push_back(it->second);
    } else {
      ids.push_back(khmer_tokenizer::UNK_ID);
    }
    p += len;
  }

  return ids;
}

std::string khmer_tokenizer_decode(
  const khmer_tokenizer & tok,
  const std::vector<int32_t> & ids
) {
  std::string result;
  for (int32_t id : ids) {
    int32_t idx = id - 4;
    if (idx >= 0 && idx < static_cast<int32_t>(tok.vocab.size())) {
      result += tok.vocab[idx];
    }
  }
  return result;
}
