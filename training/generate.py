import os
import regex as re
from khmersegment import Segmenter
from nltk.tokenize import TweetTokenizer
from khnormal import khnormal

tknzr = TweetTokenizer(reduce_len=True, strip_handles=False)

segmenter = Segmenter("-m assets/km-5tag-seg-model")
re_pre_segment = re.compile(
  r"([\u1780-\u17dd]+)|([\u17e0-\u17e90-9]+)|([^\u1780-\u17ff]+)"
)


def segment(text: str):
  for m in re_pre_segment.finditer(text):
    if m[2]:
      yield m[2]
      continue
    if m[1]:
      for segment in segmenter(m[1], deep=True):
        yield segment
      continue

    if len(m[0].strip()) == 0:
      yield m[0]
      continue

    tokens = tknzr.tokenize(m[0])
    if len(tokens) == 0:
      yield m[0]
      continue
    yield from tokens


if __name__ == "__main__":
  text_path = "/Users/seanghay/Projects/github/khmer-text-crawler/train.txt"

  os.makedirs("data", exist_ok=True)
  c = 0
  with open("data/train.txt", "w") as outfile:
    with open(text_path) as infile:
      for line in infile:
        line = line.rstrip("\n")
        #print(line)
        line = khnormal(line)
        for s in segment(line):
          c += 1
          outfile.write(s + "\n")
          print(c)
        if c > 10_000_000:
          break
