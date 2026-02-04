#include "tokenizer.h"
#include "utf8.h"

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

void khmer_tokenizer_init(khmer_tokenizer & tok) {
  tok.vocab.clear();
  tok.codepoint_to_id.clear();

  for (size_t i = 0; i < VOCAB_SIZE; i++) {
    tok.vocab.push_back(VOCAB[i]);
    const char * it = VOCAB[i];
    uint32_t cp = utf8::unchecked::next(it);
    tok.codepoint_to_id[cp] = static_cast<int32_t>(i + 4);
  }
}

std::vector<int32_t> khmer_tokenizer_encode(
  const khmer_tokenizer & tok,
  const char * text
) {
  std::vector<int32_t> ids;
  const char * p = text;

  while (*p) {
    uint32_t cp = utf8::unchecked::next(p);

    auto it = tok.codepoint_to_id.find(cp);
    if (it != tok.codepoint_to_id.end()) {
      ids.push_back(it->second);
    } else {
      ids.push_back(khmer_tokenizer::UNK_ID);
    }
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
