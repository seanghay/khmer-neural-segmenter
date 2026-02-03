#ifndef KHMER_CRF_H
#define KHMER_CRF_H

#include <cstdint>
#include <vector>

// CRF Viterbi decoding
// emissions: [seq_len * num_labels] flattened emission scores
// Returns: best tag sequence
std::vector<int32_t> crf_viterbi_decode(
  const float * emissions,
  int seq_len,
  int num_labels,
  const float * start_trans,
  const float * end_trans,
  const float * transitions
);

#endif // KHMER_CRF_H
