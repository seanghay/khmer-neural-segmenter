#include "khmer-segmenter.h"
#include "crf.h"
#include "model_data.h"
#include "ggml.h"
#include "gguf.h"
#include "utf8.h"

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

// Buffer reader for parsing GGUF binary format directly from memory
struct gguf_buf_reader {
  const uint8_t * buf;
  size_t size;
  size_t offset;

  gguf_buf_reader(const void * data, size_t len)
    : buf(static_cast<const uint8_t *>(data)), size(len), offset(0) {}

  void read_raw(void * dst, size_t n) {
    memcpy(dst, buf + offset, n);
    offset += n;
  }

  uint32_t read_u32() { uint32_t v; read_raw(&v, 4); return v; }
  uint64_t read_u64() { uint64_t v; read_raw(&v, 8); return v; }
  int64_t  read_i64() { int64_t  v; read_raw(&v, 8); return v; }

  std::string read_string() {
    uint64_t len = read_u64();
    std::string s(reinterpret_cast<const char *>(buf + offset), len);
    offset += len;
    return s;
  }

  void skip(size_t n) { offset += n; }
  void align_to(size_t alignment) {
    size_t rem = offset % alignment;
    if (rem) offset += alignment - rem;
  }
};

// Parsed tensor info from GGUF header
struct parsed_tensor_info {
  std::string name;
  uint32_t n_dims;
  int64_t  ne[GGML_MAX_DIMS];
  uint32_t type;
  uint64_t offset;
};

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

// Load model from memory buffer by parsing GGUF directly
struct khmer_context * khmer_init_from_buffer(const void * data, size_t size) {
  gguf_buf_reader r(data, size);

  // Validate magic
  uint32_t magic = r.read_u32();
  if (memcmp(&magic, "GGUF", 4) != 0) {
    fprintf(stderr, "Invalid GGUF magic\n");
    return nullptr;
  }

  // Read header
  uint32_t version     = r.read_u32();
  uint64_t n_tensors   = r.read_u64();
  uint64_t n_kv        = r.read_u64();

  if (version != 3) {
    fprintf(stderr, "Unsupported GGUF version: %u\n", version);
    return nullptr;
  }

  // Parse KV pairs, extracting hparams
  khmer_hparams hparams;
  for (uint64_t i = 0; i < n_kv; i++) {
    std::string key = r.read_string();
    uint32_t vtype = r.read_u32();

    if (vtype == 4 /* UINT32 */) {
      uint32_t val = r.read_u32();
      if (key == "khmer.vocab_size")     hparams.vocab_size     = val;
      else if (key == "khmer.embedding_dim") hparams.embedding_dim = val;
      else if (key == "khmer.hidden_dim")    hparams.hidden_dim    = val;
      else if (key == "khmer.num_labels")    hparams.num_labels    = val;
    } else if (vtype == 8 /* STRING */) {
      uint64_t slen = r.read_u64();
      r.skip(slen);
    } else if (vtype == 9 /* ARRAY */) {
      uint32_t arr_type = r.read_u32();
      uint64_t arr_len  = r.read_u64();
      // Skip array elements
      static const size_t type_sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
      if (arr_type == 8) {
        for (uint64_t j = 0; j < arr_len; j++) {
          uint64_t sl = r.read_u64();
          r.skip(sl);
        }
      } else if (arr_type < 13 && arr_type != 9) {
        r.skip(arr_len * type_sizes[arr_type]);
      }
    } else {
      // Scalar types: skip by size
      static const size_t type_sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
      if (vtype < 13) {
        r.skip(type_sizes[vtype]);
      }
    }
  }

  // Parse tensor info entries
  std::vector<parsed_tensor_info> tensors(n_tensors);
  for (uint64_t i = 0; i < n_tensors; i++) {
    tensors[i].name   = r.read_string();
    tensors[i].n_dims = r.read_u32();
    for (int d = 0; d < GGML_MAX_DIMS; d++) tensors[i].ne[d] = 1;
    for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
      tensors[i].ne[d] = static_cast<int64_t>(r.read_u64());
    }
    tensors[i].type   = r.read_u32();
    tensors[i].offset = r.read_u64();
  }

  // Data section starts at current position, aligned to 32 bytes
  size_t data_offset = GGML_PAD(r.offset, 32);

  // Compute total tensor data size for ggml_init
  size_t total_size = 0;
  for (auto & ti : tensors) {
    size_t row_size = ggml_row_size(static_cast<enum ggml_type>(ti.type), ti.ne[0]);
    size_t n_elements = 1;
    for (uint32_t d = 1; d < ti.n_dims; d++) n_elements *= ti.ne[d];
    total_size += GGML_PAD(row_size * n_elements, GGML_MEM_ALIGN);
  }

  // Create ggml_context with enough memory for all tensors + overhead
  struct ggml_init_params params;
  params.mem_size   = total_size + n_tensors * ggml_tensor_overhead();
  params.mem_buffer = nullptr;
  params.no_alloc   = false;

  struct ggml_context * ctx_w = ggml_init(params);
  if (!ctx_w) {
    fprintf(stderr, "Failed to allocate ggml context\n");
    return nullptr;
  }

  // Create tensors and copy data from buffer
  const uint8_t * buf = static_cast<const uint8_t *>(data);
  for (auto & ti : tensors) {
    struct ggml_tensor * t = ggml_new_tensor(ctx_w,
      static_cast<enum ggml_type>(ti.type), ti.n_dims, ti.ne);
    ggml_set_name(t, ti.name.c_str());
    memcpy(t->data, buf + data_offset + ti.offset, ggml_nbytes(t));
  }

  // Build khmer_context
  auto * ctx = new khmer_context();
  ctx->model.ctx_w    = ctx_w;
  ctx->model.gguf_ctx = nullptr;
  ctx->model.hparams  = hparams;

  // Load tensors by name
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

  int num_labels = hparams.num_labels;
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
  int32_t prev_tag = -1;

  // Iterate through text characters, skipping BOS (index 0) and EOS (last)
  const char * p = text;
  size_t pred_idx = 1;  // start at 1 to skip BOS
  size_t pred_end = predictions.size() - 1;  // stop before EOS

  while (*p && pred_idx < pred_end) {
    // Get UTF-8 character
    int len = utf8::internal::sequence_length(p);

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
      if (prev_tag != 0 && !current_word.empty()) {
        result.push_back(std::move(current_word));
        current_word.clear();
      }
      current_word += ch;
    }

    prev_tag = tag;
    p += len;
    pred_idx++;
  }

  // Flush remaining word
  if (!current_word.empty()) {
    result.push_back(std::move(current_word));
  }

  return result;
}
