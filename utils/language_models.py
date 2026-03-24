"""
Language model implementations:
  - UnigramModel   (MLE + Laplace smoothing)
  - BigramModel    (MLE + Laplace + Good-Turing + Kneser-Ney)
  - Shared utilities: log-probability, perplexity
"""

import math
import collections
from typing import List, Dict, Tuple, Optional


# ─── Unigram Model ────────────────────────────────────────────────────────────

class UnigramModel:
    """
    Unigram language model with Maximum Likelihood Estimation
    and optional Laplace (add-k) smoothing.
    """

    def __init__(self, smoothing: str = 'none', k: float = 1.0):
        """
        Args:
            smoothing: 'none' | 'laplace'
            k: additive smoothing constant (default 1 for Laplace)
        """
        self.smoothing = smoothing
        self.k = k
        self.counts: Dict[str, int] = {}
        self.total: int = 0
        self.vocab_size: int = 0

    def fit(self, corpus: List[List[str]]):
        """Train on tokenised corpus (list of token lists)."""
        self.counts = collections.Counter(tok for sent in corpus for tok in sent)
        self.total = sum(self.counts.values())
        self.vocab_size = len(self.counts)
        return self

    def prob(self, word: str) -> float:
        c = self.counts.get(word, 0)
        if self.smoothing == 'laplace':
            return (c + self.k) / (self.total + self.k * self.vocab_size)
        return c / max(self.total, 1)

    def log_prob(self, word: str) -> float:
        p = self.prob(word)
        return math.log(p) if p > 0 else float('-inf')

    def sentence_log_prob(self, tokens: List[str]) -> float:
        return sum(self.log_prob(t) for t in tokens)

    def perplexity(self, corpus: List[List[str]]) -> float:
        tokens = [t for sent in corpus for t in sent]
        log_prob_sum = sum(self.log_prob(t) for t in tokens)
        N = len(tokens)
        return math.exp(-log_prob_sum / max(N, 1))

    def top_n(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted([(w, self.prob(w)) for w in self.counts],
                      key=lambda x: -x[1])[:n]


# ─── Bigram Model ─────────────────────────────────────────────────────────────

class BigramModel:
    """
    Bigram language model with multiple smoothing options:
      - 'none'       : MLE
      - 'laplace'    : Add-k smoothing
      - 'good_turing': Simple Good-Turing approximation
      - 'kneser_ney' : Kneser-Ney smoothing
    """

    BOS = '<s>'
    EOS = '</s>'

    def __init__(self, smoothing: str = 'laplace', k: float = 1.0, discount: float = 0.75):
        self.smoothing = smoothing
        self.k = k
        self.discount = discount
        self.unigram_counts: Dict[str, int] = {}
        self.bigram_counts: Dict[Tuple[str, str], int] = {}
        self.vocab: set = set()
        self.total_unigrams: int = 0

    def _pad(self, tokens: List[str]) -> List[str]:
        return [self.BOS] + tokens + [self.EOS]

    def fit(self, corpus: List[List[str]]):
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()

        for sent in corpus:
            padded = self._pad(sent)
            self.unigram_counts.update(padded)
            self.bigram_counts.update(zip(padded[:-1], padded[1:]))

        self.vocab = set(self.unigram_counts.keys())
        self.total_unigrams = sum(self.unigram_counts.values())

        if self.smoothing == 'good_turing':
            self._fit_good_turing()
        elif self.smoothing == 'kneser_ney':
            self._fit_kneser_ney()
        return self

    # ── Good-Turing helpers ───────────────────────────────────────────────────
    def _fit_good_turing(self):
        freq_of_freq = collections.Counter(self.bigram_counts.values())
        self._gt_freq_of_freq = freq_of_freq
        self._N = sum(self.bigram_counts.values())
        # Number of unique words following each context (needed for unseen mass)
        self._n1plus_after: Dict[str, int] = collections.defaultdict(int)
        for (w1, _) in self.bigram_counts:
            self._n1plus_after[w1] += 1

    def _gt_count(self, c: int) -> float:
        """Good-Turing adjusted count."""
        fof = self._gt_freq_of_freq
        N_c = fof.get(c, 0)
        N_c1 = fof.get(c + 1, 0)
        if N_c == 0 or N_c1 == 0:
            return c
        return (c + 1) * N_c1 / N_c

    # ── Kneser-Ney helpers ────────────────────────────────────────────────────
    def _fit_kneser_ney(self):
        # Continuation count: number of unique preceding contexts for word w
        word_contexts: Dict[str, set] = collections.defaultdict(set)
        for (w1, w2) in self.bigram_counts:
            word_contexts[w2].add(w1)
        self._kn_continuation = {w: len(ctx) for w, ctx in word_contexts.items()}
        self._kn_total = sum(self._kn_continuation.values())

        # N1+ (w •) — number of unique words following w1
        self._n1plus_after: Dict[str, int] = collections.defaultdict(int)
        for (w1, _) in self.bigram_counts:
            self._n1plus_after[w1] += 1

    def _kn_prob_lower(self, word: str) -> float:
        """Kneser-Ney lower-order (continuation) probability."""
        return self._kn_continuation.get(word, 0) / max(self._kn_total, 1)

    # ── Core probability ─────────────────────────────────────────────────────

    def prob(self, word: str, context: str) -> float:
        c_bigram = self.bigram_counts.get((context, word), 0)
        c_context = self.unigram_counts.get(context, 0)
        V = len(self.vocab)

        if self.smoothing == 'none':
            if c_context == 0:
                return 0.0
            return c_bigram / c_context

        elif self.smoothing == 'laplace':
            return (c_bigram + self.k) / (c_context + self.k * V)

        elif self.smoothing == 'good_turing':
            # Simple Good-Turing: use adjusted count normalized by context count
            # Fall back to Laplace when GT adjustment is unstable (c=0 or N_c+1=0)
            adj = self._gt_count(c_bigram)
            if c_context == 0:
                # back-off to uniform over vocab
                return 1.0 / max(V, 1)
            # For seen bigrams use GT-adjusted count; for unseen reserve a
            # probability mass proportional to N1 / N (number of singletons)
            if c_bigram == 0:
                # Unseen: allocate probability mass from singleton mass
                n1 = self._gt_freq_of_freq.get(1, 1)
                reserved = n1 / max(self._N, 1)
                n_unseen = max(V - self._n1plus_after.get(context, 0), 1)
                return reserved / n_unseen
            return adj / max(c_context, 1)

        elif self.smoothing == 'kneser_ney':
            d = self.discount
            numerator = max(c_bigram - d, 0)
            if c_context == 0:
                return self._kn_prob_lower(word) or (1.0 / max(V, 1))
            n1plus = max(self._n1plus_after.get(context, 0), 0)
            lam = (d * n1plus) / max(c_context, 1)
            lower = self._kn_prob_lower(word) or (1.0 / max(V, 1))
            return numerator / c_context + lam * lower

        return 0.0

    def log_prob(self, word: str, context: str) -> float:
        p = self.prob(word, context)
        return math.log(p) if p > 0 else float('-inf')

    def sentence_log_prob(self, tokens: List[str]) -> float:
        padded = self._pad(tokens)
        return sum(self.log_prob(w2, w1) for w1, w2 in zip(padded[:-1], padded[1:]))

    def perplexity(self, corpus: List[List[str]]) -> float:
        log_prob_total = 0.0
        N = 0
        for sent in corpus:
            padded = self._pad(sent)
            N += len(padded) - 1
            for w1, w2 in zip(padded[:-1], padded[1:]):
                lp = self.log_prob(w2, w1)
                if lp == float('-inf'):
                    log_prob_total += -100  # penalty
                else:
                    log_prob_total += lp
        return math.exp(-log_prob_total / max(N, 1))

    def generate(self, seed: str = None, max_len: int = 20) -> List[str]:
        """Generate text by sampling from bigram distribution."""
        import random
        context = seed if seed else self.BOS
        result = [context] if context != self.BOS else []

        for _ in range(max_len):
            candidates = {w2: c for (w1, w2), c in self.bigram_counts.items()
                          if w1 == context}
            if not candidates:
                break
            words, weights = zip(*candidates.items())
            total = sum(weights)
            probs = [w / total for w in weights]
            next_word = random.choices(words, probs)[0]
            if next_word == self.EOS:
                break
            result.append(next_word)
            context = next_word
        return result


# ─── Entropy Utilities ────────────────────────────────────────────────────────

def entropy(prob_dist: Dict[str, float]) -> float:
    """Shannon entropy H(X) = -Σ p(x) log2 p(x)."""
    return -sum(p * math.log2(p) for p in prob_dist.values() if p > 0)


def cross_entropy(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """Cross-entropy H(P, Q) = -Σ p(x) log2 q(x)."""
    result = 0.0
    for word, p in p_dist.items():
        q = q_dist.get(word, 1e-10)
        result -= p * math.log2(q)
    return result


def kl_divergence(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """KL divergence D_KL(P || Q) = Σ p(x) log2(p(x)/q(x))."""
    result = 0.0
    for word, p in p_dist.items():
        q = q_dist.get(word, 1e-10)
        if p > 0:
            result += p * math.log2(p / q)
    return result


def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {w: c / total for w, c in counts.items()}
