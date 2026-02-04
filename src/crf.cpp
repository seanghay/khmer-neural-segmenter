#include "crf.h"
#include <cmath>
#include <algorithm>
#include <limits>

std::vector<int32_t> crf_viterbi_decode(
  const float * emissions,
  int seq_len,
  int num_labels,
  const float * start_trans,
  const float * end_trans,
  const float * transitions
) {
  if (seq_len == 0) {
    return {};
  }

  // Initialize with start transitions + first emission
  std::vector<float> score(num_labels);
  for (int j = 0; j < num_labels; j++) {
    score[j] = start_trans[j] + emissions[j];
  }

  // Pre-allocate working memory
  std::vector<float> next_score(num_labels);
  std::vector<int32_t> history((seq_len - 1) * num_labels);

  // Forward pass
  for (int t = 1; t < seq_len; t++) {
    for (int j = 0; j < num_labels; j++) {
      float best = -std::numeric_limits<float>::infinity();
      int32_t best_i = 0;

      for (int i = 0; i < num_labels; i++) {
        float s = score[i] + transitions[i * num_labels + j]
                + emissions[t * num_labels + j];
        if (s > best) {
          best = s;
          best_i = i;
        }
      }
      next_score[j] = best;
      history[(t - 1) * num_labels + j] = best_i;
    }
    std::swap(score, next_score);
  }

  // Add end transitions
  for (int j = 0; j < num_labels; j++) {
    score[j] += end_trans[j];
  }

  // Find best final state
  int32_t best_last = 0;
  float best_score = score[0];
  for (int j = 1; j < num_labels; j++) {
    if (score[j] > best_score) {
      best_score = score[j];
      best_last = j;
    }
  }

  // Backtrack
  std::vector<int32_t> best_path(seq_len);
  best_path[seq_len - 1] = best_last;

  for (int t = seq_len - 2; t >= 0; t--) {
    best_path[t] = history[t * num_labels + best_path[t + 1]];
  }

  return best_path;
}
