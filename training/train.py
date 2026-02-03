import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import Segmenter
from data import TextDataset, collate_fn
from tokenizer import Tokenizer
from tqdm import tqdm


def train():
  device = "mps"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  tokenizer = Tokenizer()

  train_dataset = TextDataset(tokenizer=tokenizer, split="train")
  eval_dataset = TextDataset(tokenizer=tokenizer, split="eval")

  train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
  )

  eval_loader = DataLoader(
    eval_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
  )

  print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

  model = Segmenter(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    hidden_dim=512,
    num_labels=3,
  )

  model.to(device)

  optimizer = AdamW(model.parameters(), lr=1e-3)
  num_epochs = 10

  for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, tags, mask) in enumerate(tqdm(train_loader, desc="Train")):
      inputs = inputs.to(device)
      tags = tags.to(device)
      mask = mask.to(device)

      optimizer.zero_grad()
      loss = model(inputs, tags=tags, mask=mask)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
      for inputs, tags, mask in tqdm(eval_loader, desc="Eval"):
        inputs = inputs.to(device)
        tags = tags.to(device)
        mask = mask.to(device)

        loss = model(inputs, tags=tags, mask=mask)
        eval_loss += loss.item()

        predictions = model(inputs, mask=mask)
        for pred, target, m in zip(predictions, tags, mask):
          for p, t, valid in zip(pred, target, m):
            if valid:
              total += 1
              if p == t.item():
                correct += 1

    avg_eval_loss = eval_loss / len(eval_loader)
    accuracy = correct / total if total > 0 else 0

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")

  torch.save(model.state_dict(), "model.pt")
  print("Model saved to model.pt")


if __name__ == "__main__":
  train()
