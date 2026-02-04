#ifndef KHMER_TOKENIZER_H
#define KHMER_TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

struct khmer_tokenizer {
  std::vector<std::string> vocab;
  std::unordered_map<uint32_t, int32_t> codepoint_to_id;

  static const int32_t PAD_ID = 0;
  static const int32_t BOS_ID = 1;
  static const int32_t EOS_ID = 2;
  static const int32_t UNK_ID = 3;
};

void khmer_tokenizer_init(khmer_tokenizer & tok);

std::vector<int32_t> khmer_tokenizer_encode(
  const khmer_tokenizer & tok,
  const char * text
);

std::string khmer_tokenizer_decode(
  const khmer_tokenizer & tok,
  const std::vector<int32_t> & ids
);

#endif // KHMER_TOKENIZER_H
