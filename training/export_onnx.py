import torch
import torch.nn as nn
import numpy as np
from model import Segmenter
from tokenizer import Tokenizer


class SegmenterEmissions(nn.Module):
  """Wrapper that outputs emissions only (for ONNX export)."""

  def __init__(self, segmenter):
    super().__init__()
    self.embedding = segmenter.embedding
    self.gru = segmenter.gru
    self.fc = segmenter.fc

  def forward(self, x):
    embedded = self.embedding(x)
    gru_out, _ = self.gru(embedded)
    emissions = self.fc(gru_out)
    return emissions


def export_to_onnx(
  model_path="best_model.pt",
  onnx_path="segmenter.onnx",
  crf_path="crf_params.npz",
):
  tokenizer = Tokenizer()
  model = Segmenter(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    hidden_dim=256,
    num_labels=3,
  )
  model.load_state_dict(torch.load(model_path, map_location="cpu"))
  model.eval()

  # Extract CRF parameters
  crf = model.crf
  print(crf.start_transitions)
  print(crf.end_transitions)
  print(crf.transitions)

  np.savez(
    crf_path,
    start_transitions=crf.start_transitions.detach().numpy(),
    end_transitions=crf.end_transitions.detach().numpy(),
    transitions=crf.transitions.detach().numpy(),
  )
  print(f"Saved CRF parameters to {crf_path}")

  # Create emissions-only model
  emissions_model = SegmenterEmissions(model)
  emissions_model.eval()

  # Create dummy input for tracing
  dummy_input = torch.randint(0, len(tokenizer), (1, 32), dtype=torch.long)

  # Export to ONNX (use legacy export to avoid dynamo issues)
  torch.onnx.export(
    emissions_model,
    dummy_input,
    onnx_path,
    input_names=["input_ids"],
    output_names=["emissions"],
    dynamic_axes={
      "input_ids": {0: "batch_size", 1: "sequence_length"},
      "emissions": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=14,
    dynamo=False,
  )
  print(f"Exported ONNX model to {onnx_path}")


def viterbi_decode(emissions, start_transitions, end_transitions, transitions):
  """Viterbi decoding for CRF inference."""
  seq_length, _ = emissions.shape

  # Initialize
  score = start_transitions + emissions[0]
  history = []

  # Forward pass
  for i in range(1, seq_length):
    broadcast_score = score.reshape(-1, 1)
    broadcast_emissions = emissions[i].reshape(1, -1)
    next_score = broadcast_score + transitions + broadcast_emissions
    indices = next_score.argmax(axis=0)
    score = next_score.max(axis=0)
    history.append(indices)

  # Add end transitions
  score += end_transitions

  # Backtrack
  best_tags = [int(score.argmax())]
  for hist in reversed(history):
    best_tags.append(int(hist[best_tags[-1]]))
  best_tags.reverse()

  return best_tags


def segment_onnx(text, session, tokenizer, crf_params):
  """Segment text using ONNX Runtime."""
  token_ids = tokenizer.encode(text)
  inputs = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]
  input_array = np.array([inputs], dtype=np.int64)

  # Run inference
  emissions = session.run(None, {"input_ids": input_array})[0][0]

  # Viterbi decode
  predictions = viterbi_decode(
    emissions,
    crf_params["start_transitions"],
    crf_params["end_transitions"],
    crf_params["transitions"],
  )

  # Remove BOS/EOS predictions
  predictions = predictions[1:-1]

  # Segment based on B-WORD (1) tags
  words = []
  current_word = []

  for char, tag in zip(text, predictions):
    if tag == 1:  # B-WORD
      if current_word:
        words.append("".join(current_word))
      current_word = [char]
    elif tag == 2:  # I-WORD
      current_word.append(char)
    else:  # 0 (non-Khmer)
      if current_word:
        words.append("".join(current_word))
        current_word = []
      words.append(char)

  if current_word:
    words.append("".join(current_word))

  return words


if __name__ == "__main__":
  # Export model
  export_to_onnx()

  # Test ONNX inference
  import onnxruntime as ort

  tokenizer = Tokenizer()
  session = ort.InferenceSession("segmenter.onnx")
  crf_params = np.load("crf_params.npz")

  text = "គិតចាប់ពី ខែធ្នូ ឆ្នាំ២០២៤ មកដល់ថ្ងៃទី១១".replace("\u200b", "")
  words = segment_onnx(text, session, tokenizer, crf_params)
  print(f"ONNX result: {'|'.join(words)}")

  # Compare with PyTorch
  from infer import load_model, segment

  model, tokenizer = load_model()
  words_pt = segment(text, model, tokenizer)
  print(f"PyTorch result: {'|'.join(words_pt)}")
