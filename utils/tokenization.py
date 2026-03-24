"""
Advanced tokenization utilities including subword tokenization,
sentence-level processing, and corpus-level statistics.
"""

import re
import collections
from typing import List, Dict, Tuple, Iterator


def build_vocab(corpus: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """Build vocabulary from tokenized corpus."""
    counter = collections.Counter(tok for sent in corpus for tok in sent)
    return {word: freq for word, freq in counter.items() if freq >= min_freq}


def get_token_frequencies(tokens: List[str]) -> Dict[str, int]:
    """Get word frequency distribution."""
    return dict(collections.Counter(tokens))


def ngrams(tokens: List[str], n: int) -> List[Tuple]:
    """Generate n-grams from token list."""
    return list(zip(*[tokens[i:] for i in range(n)]))


def bigrams(tokens: List[str]) -> List[Tuple]:
    return ngrams(tokens, 2)


def trigrams(tokens: List[str]) -> List[Tuple]:
    return ngrams(tokens, 3)


def sentence_to_tokens(sentence: str, lowercase: bool = True) -> List[str]:
    """Basic regex tokenizer."""
    if lowercase:
        sentence = sentence.lower()
    return re.findall(r"\b[a-zA-Z']+\b", sentence)


def corpus_stats(corpus: List[List[str]]) -> Dict:
    """Compute basic corpus statistics."""
    all_tokens = [tok for sent in corpus for tok in sent]
    vocab = set(all_tokens)
    return {
        'num_sentences': len(corpus),
        'num_tokens': len(all_tokens),
        'vocab_size': len(vocab),
        'avg_sent_length': len(all_tokens) / max(len(corpus), 1),
        'type_token_ratio': len(vocab) / max(len(all_tokens), 1),
    }


def pad_sequence(tokens: List[str], n: int,
                 bos: str = '<s>', eos: str = '</s>') -> List[str]:
    """Add BOS/EOS padding for n-gram models."""
    return [bos] * (n - 1) + tokens + [eos]


def sliding_window(tokens: List[str], window_size: int, step: int = 1) -> Iterator[List[str]]:
    """Generate sliding windows over tokens."""
    for i in range(0, len(tokens) - window_size + 1, step):
        yield tokens[i:i + window_size]


class SimpleTokenizer:
    """Simple word tokenizer with vocabulary management."""

    def __init__(self, min_freq: int = 1, max_vocab: int = None,
                 unk_token: str = '<UNK>', pad_token: str = '<PAD>'):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {}
        self.idx2word = {}
        self.frequencies = {}

    def fit(self, corpus: List[List[str]]):
        """Build vocabulary from corpus."""
        counter = collections.Counter(tok for sent in corpus for tok in sent)
        self.frequencies = dict(counter)

        # Filter by min frequency
        filtered = [(w, c) for w, c in counter.most_common() if c >= self.min_freq]

        # Limit vocab size
        if self.max_vocab:
            filtered = filtered[:self.max_vocab - 2]

        special = [self.pad_token, self.unk_token]
        vocab = special + [w for w, _ in filtered]

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        unk_idx = self.word2idx.get(self.unk_token, 0)
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.unk_token) for i in indices]

    @property
    def vocab_size(self):
        return len(self.word2idx)
