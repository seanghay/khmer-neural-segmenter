#ifndef KHMER_SEGMENTER_H
#define KHMER_SEGMENTER_H

#include "tokenizer.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct khmer_hparams {
  int32_t vocab_size   = 182;  // 178 chars + 4 special tokens
  int32_t embedding_dim = 256;
  int32_t hidden_dim   = 256;
  int32_t num_labels   = 3;
};

struct khmer_model {
  khmer_hparams hparams;

  // Embedding
  struct ggml_tensor * embedding;  // [vocab_size, embedding_dim]

  // BiGRU forward
  struct ggml_tensor * gru_w_ih_f;   // [3*hidden, embedding]
  struct ggml_tensor * gru_w_hh_f;   // [3*hidden, hidden]
  struct ggml_tensor * gru_b_ih_f;   // [3*hidden]
  struct ggml_tensor * gru_b_hh_f;   // [3*hidden]

  // BiGRU backward
  struct ggml_tensor * gru_w_ih_b;
  struct ggml_tensor * gru_w_hh_b;
  struct ggml_tensor * gru_b_ih_b;
  struct ggml_tensor * gru_b_hh_b;

  // Output projection
  struct ggml_tensor * fc_w;         // [num_labels, 2*hidden]
  struct ggml_tensor * fc_b;         // [num_labels]

  // CRF parameters (stored separately as they're used in CPU decoding)
  std::vector<float> crf_start;      // [num_labels]
  std::vector<float> crf_end;        // [num_labels]
  std::vector<float> crf_trans;      // [num_labels * num_labels]

  // GGML context for weights
  struct ggml_context * ctx_w;
};

struct khmer_context {
  khmer_model model;
  khmer_tokenizer tokenizer;
};

// Load model from embedded data (no file needed)
struct khmer_context * khmer_init();

// Load model from memory buffer
struct khmer_context * khmer_init_from_buffer(const void * data, size_t size);

// Load model from GGUF file
struct khmer_context * khmer_init_from_file(const char * path);

// Free resources
void khmer_free(struct khmer_context * ctx);

// Segment text, returns a vector of word strings
std::vector<std::string> khmer_segment(struct khmer_context * ctx, const char * text);

#endif // KHMER_SEGMENTER_H
