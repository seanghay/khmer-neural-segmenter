#include "khmer-segmenter.h"
#include "crf.h"
#include "model_data.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#define fdopen _fdopen
#define close  _close
#define unlink _unlink
#else
#include <unistd.h>
#endif

// Helper: sigmoid
static float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Helper: matrix-vector multiply: y = W @ x + b
// W: [out_dim, in_dim], x: [in_dim], b: [out_dim], y: [out_dim]
static void matvec_add(
  float * y,
  const float * W,
  const float * x,
  const float * b,
  int out_dim,
  int in_dim
) {
  for (int i = 0; i < out_dim; i++) {
    float sum = b ? b[i] : 0.0f;
    for (int j = 0; j < in_dim; j++) {
      sum += W[i * in_dim + j] * x[j];
    }
    y[i] = sum;
  }
}

// GRU cell: compute h_new from x and h_prev
// W_ih: [3*H, I], W_hh: [3*H, H], b_ih: [3*H], b_hh: [3*H]
// Gates order: r, z, n (reset, update, new)
// scratch_gates_x and scratch_gates_h must be pre-allocated to at least 3*hidden_dim floats
static void gru_cell(
  float * h_new,
  const float * x,
  const float * h,
  const float * W_ih,
  const float * W_hh,
  const float * b_ih,
  const float * b_hh,
  int hidden_dim,
  int input_dim,
  float * scratch_gates_x,
  float * scratch_gates_h
) {
  int H = hidden_dim;
  int H3 = 3 * hidden_dim;

  // Compute gates: W_ih @ x + b_ih and W_hh @ h + b_hh
  matvec_add(scratch_gates_x, W_ih, x, b_ih, H3, input_dim);
  matvec_add(scratch_gates_h, W_hh, h, b_hh, H3, H);

  // Fused gate computation: r, z, n, h_new in a single loop
  for (int i = 0; i < H; i++) {
    float ri = sigmoid(scratch_gates_x[i] + scratch_gates_h[i]);
    float zi = sigmoid(scratch_gates_x[H + i] + scratch_gates_h[H + i]);
    float ni = tanhf(scratch_gates_x[2*H + i] + ri * scratch_gates_h[2*H + i]);
    h_new[i] = (1.0f - zi) * ni + zi * h[i];
  }
}

// Common initialization from gguf context
static struct khmer_context * khmer_init_from_gguf(struct gguf_context * gguf_ctx, struct ggml_context * ctx_w) {
  if (!gguf_ctx || !ctx_w) {
    return nullptr;
  }

  // Allocate context
  auto * ctx = new khmer_context();
  ctx->model.ctx_w = ctx_w;
  ctx->model.gguf_ctx = gguf_ctx;

  // Read hyperparameters from metadata
  int key_idx;

  key_idx = gguf_find_key(gguf_ctx, "khmer.vocab_size");
  if (key_idx >= 0) {
    ctx->model.hparams.vocab_size = gguf_get_val_u32(gguf_ctx, key_idx);
  }

  key_idx = gguf_find_key(gguf_ctx, "khmer.embedding_dim");
  if (key_idx >= 0) {
    ctx->model.hparams.embedding_dim = gguf_get_val_u32(gguf_ctx, key_idx);
  }

  key_idx = gguf_find_key(gguf_ctx, "khmer.hidden_dim");
  if (key_idx >= 0) {
    ctx->model.hparams.hidden_dim = gguf_get_val_u32(gguf_ctx, key_idx);
  }

  key_idx = gguf_find_key(gguf_ctx, "khmer.num_labels");
  if (key_idx >= 0) {
    ctx->model.hparams.num_labels = gguf_get_val_u32(gguf_ctx, key_idx);
  }

  // Load tensors
  ctx->model.embedding   = ggml_get_tensor(ctx_w, "embedding.weight");
  ctx->model.gru_w_ih_f  = ggml_get_tensor(ctx_w, "gru.weight_ih_l0");
  ctx->model.gru_w_hh_f  = ggml_get_tensor(ctx_w, "gru.weight_hh_l0");
  ctx->model.gru_b_ih_f  = ggml_get_tensor(ctx_w, "gru.bias_ih_l0");
  ctx->model.gru_b_hh_f  = ggml_get_tensor(ctx_w, "gru.bias_hh_l0");
  ctx->model.gru_w_ih_b  = ggml_get_tensor(ctx_w, "gru.weight_ih_l0_reverse");
  ctx->model.gru_w_hh_b  = ggml_get_tensor(ctx_w, "gru.weight_hh_l0_reverse");
  ctx->model.gru_b_ih_b  = ggml_get_tensor(ctx_w, "gru.bias_ih_l0_reverse");
  ctx->model.gru_b_hh_b  = ggml_get_tensor(ctx_w, "gru.bias_hh_l0_reverse");
  ctx->model.fc_w        = ggml_get_tensor(ctx_w, "fc.weight");
  ctx->model.fc_b        = ggml_get_tensor(ctx_w, "fc.bias");

  // Load CRF parameters
  struct ggml_tensor * crf_start = ggml_get_tensor(ctx_w, "crf.start_transitions");
  struct ggml_tensor * crf_end   = ggml_get_tensor(ctx_w, "crf.end_transitions");
  struct ggml_tensor * crf_trans = ggml_get_tensor(ctx_w, "crf.transitions");

  int num_labels = ctx->model.hparams.num_labels;
  ctx->model.crf_start.resize(num_labels);
  ctx->model.crf_end.resize(num_labels);
  ctx->model.crf_trans.resize(num_labels * num_labels);

  memcpy(ctx->model.crf_start.data(), ggml_get_data_f32(crf_start), num_labels * sizeof(float));
  memcpy(ctx->model.crf_end.data(), ggml_get_data_f32(crf_end), num_labels * sizeof(float));
  memcpy(ctx->model.crf_trans.data(), ggml_get_data_f32(crf_trans), num_labels * num_labels * sizeof(float));

  // Initialize tokenizer
  khmer_tokenizer_init(ctx->tokenizer);

  return ctx;
}

// Load model from memory buffer by writing to temp file
struct khmer_context * khmer_init_from_buffer(const void * data, size_t size) {
  // Create a temporary file
  char temp_path[256];

#ifdef _WIN32
  char temp_dir[MAX_PATH];
  GetTempPathA(MAX_PATH, temp_dir);
  snprintf(temp_path, sizeof(temp_path), "%skhmerns_model_XXXXXX", temp_dir);
  int fd = _mktemp_s(temp_path, sizeof(temp_path)) == 0 ? _open(temp_path, _O_CREAT | _O_RDWR, 0600) : -1;
#else
  snprintf(temp_path, sizeof(temp_path), "/tmp/khmerns_model_XXXXXX");
  int fd = mkstemp(temp_path);
#endif

  if (fd < 0) {
    fprintf(stderr, "Failed to create temporary file\n");
    return nullptr;
  }

  // Write the embedded data to the temp file
  FILE * fp = fdopen(fd, "wb");
  if (!fp) {
    close(fd);
    unlink(temp_path);
    fprintf(stderr, "Failed to open temporary file for writing\n");
    return nullptr;
  }

  size_t written = fwrite(data, 1, size, fp);
  fclose(fp);

  if (written != size) {
    unlink(temp_path);
    fprintf(stderr, "Failed to write model data to temporary file\n");
    return nullptr;
  }

  // Load from the temp file
  struct khmer_context * ctx = khmer_init_from_file(temp_path);

  // Delete the temp file
  unlink(temp_path);

  return ctx;
}

// Load model from embedded data
struct khmer_context * khmer_init() {
  return khmer_init_from_buffer(ggml_khmer_segmenter_gguf, ggml_khmer_segmenter_gguf_len);
}

// Load model from GGUF file
struct khmer_context * khmer_init_from_file(const char * path) {
  struct ggml_context * ctx_w = nullptr;

  struct gguf_init_params params;
  params.no_alloc = false;
  params.ctx      = &ctx_w;

  struct gguf_context * gguf_ctx = gguf_init_from_file(path, params);
  if (!gguf_ctx) {
    fprintf(stderr, "Failed to load GGUF file: %s\n", path);
    return nullptr;
  }

  if (!ctx_w) {
    fprintf(stderr, "Failed to allocate GGML context from GGUF\n");
    gguf_free(gguf_ctx);
    return nullptr;
  }

  return khmer_init_from_gguf(gguf_ctx, ctx_w);
}

void khmer_free(struct khmer_context * ctx) {
  if (ctx) {
    if (ctx->model.ctx_w) {
      ggml_free(ctx->model.ctx_w);
    }
    if (ctx->model.gguf_ctx) {
      gguf_free(ctx->model.gguf_ctx);
    }
    delete ctx;
  }
}

std::vector<std::string> khmer_segment(struct khmer_context * ctx, const char * text) {
  if (!ctx || !text) {
    return {};
  }

  const auto & model = ctx->model;
  const auto & hparams = model.hparams;

  int embed_dim = hparams.embedding_dim;
  int hidden_dim = hparams.hidden_dim;
  int num_labels = hparams.num_labels;

  // Tokenize
  std::vector<int32_t> token_ids = khmer_tokenizer_encode(ctx->tokenizer, text);

  // Add BOS/EOS (#8: reserve capacity)
  std::vector<int32_t> input_ids;
  input_ids.reserve(token_ids.size() + 2);
  input_ids.push_back(khmer_tokenizer::BOS_ID);
  input_ids.insert(input_ids.end(), token_ids.begin(), token_ids.end());
  input_ids.push_back(khmer_tokenizer::EOS_ID);

  int seq_len = static_cast<int>(input_ids.size());

  // Get embedding data
  const float * embed_data = ggml_get_data_f32(model.embedding);

  // Collect embeddings: [seq_len, embed_dim]
  std::vector<float> embeddings(seq_len * embed_dim);
  for (int t = 0; t < seq_len; t++) {
    int32_t id = input_ids[t];
    if (id < 0 || id >= hparams.vocab_size) {
      id = khmer_tokenizer::UNK_ID;
    }
    memcpy(&embeddings[t * embed_dim], &embed_data[id * embed_dim], embed_dim * sizeof(float));
  }

  // Get weight data pointers
  const float * w_ih_f = ggml_get_data_f32(model.gru_w_ih_f);
  const float * w_hh_f = ggml_get_data_f32(model.gru_w_hh_f);
  const float * b_ih_f = ggml_get_data_f32(model.gru_b_ih_f);
  const float * b_hh_f = ggml_get_data_f32(model.gru_b_hh_f);
  const float * w_ih_b = ggml_get_data_f32(model.gru_w_ih_b);
  const float * w_hh_b = ggml_get_data_f32(model.gru_w_hh_b);
  const float * b_ih_b = ggml_get_data_f32(model.gru_b_ih_b);
  const float * b_hh_b = ggml_get_data_f32(model.gru_b_hh_b);
  const float * fc_w = ggml_get_data_f32(model.fc_w);
  const float * fc_b = ggml_get_data_f32(model.fc_b);

  // Pre-allocate GRU scratch buffers (#1: avoid heap allocs in gru_cell)
  int H3 = 3 * hidden_dim;
  std::vector<float> scratch_gates_x(H3);
  std::vector<float> scratch_gates_h(H3);

  // Flat contiguous output buffers (#3)
  std::vector<float> outputs_f(seq_len * hidden_dim, 0.0f);
  std::vector<float> outputs_b(seq_len * hidden_dim, 0.0f);

  // Forward GRU pass (#2: write directly to flat buffer)
  std::vector<float> h_f(hidden_dim, 0.0f);
  for (int t = 0; t < seq_len; t++) {
    float * out = &outputs_f[t * hidden_dim];
    gru_cell(out, &embeddings[t * embed_dim], h_f.data(),
             w_ih_f, w_hh_f, b_ih_f, b_hh_f, hidden_dim, embed_dim,
             scratch_gates_x.data(), scratch_gates_h.data());
    memcpy(h_f.data(), out, hidden_dim * sizeof(float));
  }

  // Backward GRU pass (#2: write directly to flat buffer)
  std::vector<float> h_b(hidden_dim, 0.0f);
  for (int t = seq_len - 1; t >= 0; t--) {
    float * out = &outputs_b[t * hidden_dim];
    gru_cell(out, &embeddings[t * embed_dim], h_b.data(),
             w_ih_b, w_hh_b, b_ih_b, b_hh_b, hidden_dim, embed_dim,
             scratch_gates_x.data(), scratch_gates_h.data());
    memcpy(h_b.data(), out, hidden_dim * sizeof(float));
  }

  // Concatenate forward and backward: [seq_len, 2*hidden_dim]
  std::vector<float> gru_out(seq_len * 2 * hidden_dim);
  for (int t = 0; t < seq_len; t++) {
    memcpy(&gru_out[t * 2 * hidden_dim], &outputs_f[t * hidden_dim], hidden_dim * sizeof(float));
    memcpy(&gru_out[t * 2 * hidden_dim + hidden_dim], &outputs_b[t * hidden_dim], hidden_dim * sizeof(float));
  }

  // Linear layer: emissions = gru_out @ fc_w.T + fc_b
  // fc_w: [num_labels, 2*hidden_dim]
  std::vector<float> emissions(seq_len * num_labels);
  for (int t = 0; t < seq_len; t++) {
    matvec_add(&emissions[t * num_labels], fc_w, &gru_out[t * 2 * hidden_dim],
               fc_b, num_labels, 2 * hidden_dim);
  }

  // CRF Viterbi decode
  std::vector<int32_t> predictions = crf_viterbi_decode(
    emissions.data(), seq_len, num_labels,
    model.crf_start.data(), model.crf_end.data(), model.crf_trans.data()
  );

  // Reconstruct words based on tags (#7: skip BOS/EOS with index offset)
  // Tag 0: non-Khmer, Tag 1: B-WORD, Tag 2: I-WORD
  std::vector<std::string> result;
  if (predictions.size() <= 2) {
    return result;
  }
  result.reserve(token_ids.size() / 4 + 1);
  std::string current_word;

  // Iterate through text characters, skipping BOS (index 0) and EOS (last)
  const char * p = text;
  size_t pred_idx = 1;  // start at 1 to skip BOS
  size_t pred_end = predictions.size() - 1;  // stop before EOS

  while (*p && pred_idx < pred_end) {
    // Get UTF-8 character
    int len = 1;
    unsigned char c = static_cast<unsigned char>(*p);
    if ((c & 0x80) == 0) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;

    std::string ch(p, len);
    int32_t tag = predictions[pred_idx];

    if (tag == 1) {  // B-WORD
      if (!current_word.empty()) {
        result.push_back(std::move(current_word));
      }
      current_word = ch;
    } else if (tag == 2) {  // I-WORD
      current_word += ch;
    } else {  // 0 (non-Khmer)
      if (!current_word.empty()) {
        result.push_back(std::move(current_word));
        current_word.clear();
      }
      result.push_back(ch);
    }

    p += len;
    pred_idx++;
  }

  // Flush remaining word
  if (!current_word.empty()) {
    result.push_back(std::move(current_word));
  }

  return result;
}
