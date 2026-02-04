from khmercut import tokenize
import torch
from model import Segmenter
from tokenizer import Tokenizer


def load_model(path, device="cpu"):
  tokenizer = Tokenizer()
  model = Segmenter(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    hidden_dim=256,
    num_labels=3,
  )
  model.load_state_dict(torch.load(path, map_location=device))
  model.to(device)
  model.eval()
  return model, tokenizer


def segment(text, model, tokenizer, device="cpu"):
  token_ids = tokenizer.encode(text)
  inputs = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]
  inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)

  with torch.no_grad():
    predictions = model(inputs)[0]

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
  device = "cpu"
  model, tokenizer = load_model("best_model.pt", device=device)
  text = "ប្រជាជនទីបេរស់នៅ​ក្រៅស្រុក ទូទាំងពិភពលោក បានចាប់ផ្តើមនីតិវិធីបោះឆ្នោត ដើម្បីជ្រើសរើសថ្នាក់ដឹកនាំរដ្ឋាភិបាលភៀស​ខ្លួន ដែលមានទីតាំងស្ថិតនៅ​ទីក្រុង Dharamsala ភាគខាងជើងប្រទេសឥណ្ឌា។ ជាជំហាន​ដំបូង ថ្ងៃទី​១កុម្ភៈ ប្រជាជនទីបេត្រូវបោះឆ្នោត តែងតាំង​​បេក្ខជន​ជាមុនសិន ហើយជំហានបន្ទាប់ នៅថ្ងៃទី​២៦មេសា គឺត្រូវសម្រេចជ្រើសរើសក្នុងចំណោមបេក្ខជន​ឈរឈ្មោះទាំងអស់។​ លទ្ធផលជាស្ថាពរចុងក្រោយ នឹងត្រូវប្រកាស​នៅថ្ងៃ​ទី​១៣ខែឧសភា​។".replace(
    "\u200b", ""
  )

  words = segment(text, model, tokenizer, device=device)
  print("|".join(words))
