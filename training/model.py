import torch
import torch.nn as nn
from torchcrf import CRF


class Segmenter(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
    super(Segmenter, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_dim * 2, num_labels)
    self.crf = CRF(num_labels, batch_first=True)

  def forward(self, x, tags=None, mask=None):
    embedded = self.embedding(x)
    gru_out, _ = self.gru(embedded)
    emissions = self.fc(gru_out)
    if tags is not None:
      log_likelihood = self.crf(emissions, tags, mask=mask, reduction="mean")
      return -log_likelihood
    else:
      return self.crf.decode(emissions, mask=mask)


if __name__ == "__main__":
  model = Segmenter(vocab_size=200, embedding_dim=256, hidden_dim=512, num_labels=5)
  input_data = torch.randint(0, 200, (4, 10)).long()
  target_tags = torch.randint(0, 5, (4, 10)).long()

  loss = model(input_data, tags=target_tags)
  loss.backward()

  with torch.no_grad():
    best_paths = model(input_data)
    print(f"Predicted Tag Sequence: {best_paths}")
