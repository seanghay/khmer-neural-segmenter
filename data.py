import random
import torch
import re
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from torch.utils.data import Dataset


def yield_chunks(data, n, s):
  for i in range(0, len(data), s):
    yield data[i : i + n]


re_khmer = re.compile(r"[\u1780-\u17ff]+")


class TextDataset(Dataset):
  def __init__(self, tokenizer: Tokenizer, split="train", train_ratio=0.9):
    super().__init__()
    self.tokenizer = tokenizer
    with open("data/train.txt") as infile:
      lines = [line.rstrip("\n") for line in infile]

    all_items = [c for c in yield_chunks(lines, 64, random.randint(1, 64))]
    split_idx = int(len(all_items) * train_ratio)

    if split == "train":
      self.items = all_items[:split_idx]
    else:
      self.items = all_items[split_idx:]

  def __len__(self):
    return len(self.items)

  def __getitem__(self, i):
    inputs = []
    tags = []
    for w in self.items[i]:
      is_khmer = re_khmer.search(w)
      token_ids = self.tokenizer.encode(w)
      for idx, token_id in enumerate(token_ids):
        inputs.append(token_id)
        if is_khmer:
          if idx == 0:
            tags.append(1)
          else:
            tags.append(2)
        else:
          tags.append(0)

    inputs = [self.tokenizer.bos_id] + inputs + [self.tokenizer.eos_id]
    tags = [0] + tags + [0]

    return torch.LongTensor(inputs), torch.LongTensor(tags)


def collate_fn(batch):
  inputs, tags = zip(*batch)
  lengths = [len(x) for x in inputs]
  max_len = max(lengths)

  padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
  padded_tags = torch.zeros(len(batch), max_len, dtype=torch.long)
  mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

  for i, (inp, tag) in enumerate(zip(inputs, tags)):
    padded_inputs[i, : lengths[i]] = inp
    padded_tags[i, : lengths[i]] = tag
    mask[i, : lengths[i]] = True

  return padded_inputs, padded_tags, mask


if __name__ == "__main__":
  dataset = TextDataset(tokenizer=Tokenizer())
  inputs, targets = dataset[1]
  # print(inputs, targets)

  