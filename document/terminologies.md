# Terminologies

### Ngrams

NLP terminology. Under the context of `DNAZEN`, it means a sequence of adjacent token ids.

### NgramEncoder

An encoder the takes token ids as input, match the ngrams with the ngram dictionary, and output the id of matched ngrams and it's corresponding position mapping.

### Main NgramEncoder

The `NgramEncoder` used during training.

### Side NgramEncoder

The `NgramEncoder` trained from very important datasets. They are used to determine `core ngrams`.

### Core Ngrams

Core ngrams are, namely, important ngrams. They are the ngrams you need to avoid to mask during mlm task.