#!/usr/bin/env python3
"""Convert PyTorch Khmer segmenter model to GGUF format."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Segmenter
from tokenizer import Tokenizer

try:
  from gguf import GGUFWriter
except ImportError:
  print("Error: gguf package not installed. Run: pip install gguf")
  sys.exit(1)


def convert_to_gguf(model_path: str, output_path: str):
  """Convert PyTorch model to GGUF format."""

  # Load tokenizer and model
  print(f"Loading model from {model_path}...")
  tokenizer = Tokenizer()
  model = Segmenter(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    hidden_dim=256,
    num_labels=3,
  )
  model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
  model.eval()

  # Create GGUF writer
  print(f"Creating GGUF file: {output_path}")
  writer = GGUFWriter(output_path, "khmer-segmenter")

  # Write metadata
  writer.add_uint32("khmer.vocab_size", len(tokenizer))
  writer.add_uint32("khmer.embedding_dim", 256)
  writer.add_uint32("khmer.hidden_dim", 256)
  writer.add_uint32("khmer.num_labels", 3)

  # Write tensors
  print("Writing tensors...")

  # Embedding: [vocab_size, embedding_dim]
  embed_weight = model.embedding.weight.detach().numpy().astype(np.float32)
  writer.add_tensor("embedding.weight", embed_weight)
  print(f"  embedding.weight: {embed_weight.shape}")

  # GRU forward weights
  # PyTorch GRU weight_ih_l0: [3*hidden, input]
  # PyTorch GRU weight_hh_l0: [3*hidden, hidden]
  gru = model.gru
  writer.add_tensor("gru.weight_ih_l0", gru.weight_ih_l0.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.weight_hh_l0", gru.weight_hh_l0.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.bias_ih_l0", gru.bias_ih_l0.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.bias_hh_l0", gru.bias_hh_l0.detach().numpy().astype(np.float32))
  print(f"  gru.weight_ih_l0: {gru.weight_ih_l0.shape}")
  print(f"  gru.weight_hh_l0: {gru.weight_hh_l0.shape}")

  # GRU backward (reverse) weights
  writer.add_tensor("gru.weight_ih_l0_reverse", gru.weight_ih_l0_reverse.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.weight_hh_l0_reverse", gru.weight_hh_l0_reverse.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.bias_ih_l0_reverse", gru.bias_ih_l0_reverse.detach().numpy().astype(np.float32))
  writer.add_tensor("gru.bias_hh_l0_reverse", gru.bias_hh_l0_reverse.detach().numpy().astype(np.float32))
  print(f"  gru.weight_ih_l0_reverse: {gru.weight_ih_l0_reverse.shape}")

  # Linear layer: [num_labels, 2*hidden]
  fc_weight = model.fc.weight.detach().numpy().astype(np.float32)
  fc_bias = model.fc.bias.detach().numpy().astype(np.float32)
  writer.add_tensor("fc.weight", fc_weight)
  writer.add_tensor("fc.bias", fc_bias)
  print(f"  fc.weight: {fc_weight.shape}")
  print(f"  fc.bias: {fc_bias.shape}")

  # CRF parameters
  crf = model.crf
  writer.add_tensor("crf.start_transitions", crf.start_transitions.detach().numpy().astype(np.float32))
  writer.add_tensor("crf.end_transitions", crf.end_transitions.detach().numpy().astype(np.float32))
  writer.add_tensor("crf.transitions", crf.transitions.detach().numpy().astype(np.float32))
  print(f"  crf.start_transitions: {crf.start_transitions.shape}")
  print(f"  crf.end_transitions: {crf.end_transitions.shape}")
  print(f"  crf.transitions: {crf.transitions.shape}")

  # Finalize
  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()

  print(f"\nGGUF model saved to: {output_path}")

  # Print file size
  size_mb = Path(output_path).stat().st_size / (1024 * 1024)
  print(f"File size: {size_mb:.2f} MB")


def main():
  parser = argparse.ArgumentParser(
    description="Convert PyTorch Khmer segmenter to GGUF format"
  )
  parser.add_argument(
    "model_path",
    type=str,
    help="Path to PyTorch model file (best_model.pt)",
  )
  parser.add_argument(
    "output_path",
    type=str,
    help="Output GGUF file path",
  )
  args = parser.parse_args()

  convert_to_gguf(args.model_path, args.output_path)


if __name__ == "__main__":
  main()
