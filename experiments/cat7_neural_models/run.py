"""
Category 7: Neural Language Models
=====================================
Aim: Implement neural-based NLP models covering spelling correction,
     CBOW, and Skip-gram word embeddings.
Objective:
  Exercise 1 — Spelling correction via n-gram + edit distance
  Exercise 2 — CBOW model (Word2Vec)
  Exercise 3 — Skip-gram model (Word2Vec)
Dataset: Brown Corpus + Reuters Corpus
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import collections
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import brown, reuters

from utils.embeddings import train_word2vec, get_word_vectors, most_similar, reduce_pca, reduce_tsne
from utils.evaluation import save_metrics

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)


def download_data():
    for r in ['brown', 'reuters', 'punkt', 'punkt_tab', 'words']:
        nltk.download(r, quiet=True)


def load_corpus(name: str, n: int = 5000) -> list:
    if name == 'brown':
        return [[t.lower() for t in s if t.isalpha() and len(t) > 1]
                for s in list(brown.sents())[:n] if len(s) > 2]
    elif name == 'reuters':
        return [[t.lower() for t in s if t.isalpha() and len(t) > 1]
                for s in list(reuters.sents())[:n] if len(s) > 2]
    return []


# ═══════════════════════════════════════════════════════════════
# EXERCISE 1: Spelling Correction
# ═══════════════════════════════════════════════════════════════

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]


class SpellingCorrector:
    """
    Noisy channel model for spelling correction:
      argmax_w P(word) * P(typo | word)
    where P(word) is the unigram LM and
    P(typo|word) is estimated via edit distance.
    """

    def __init__(self, corpus: list, max_edit_dist: int = 2):
        self.max_edit_dist = max_edit_dist
        # Build unigram model
        counts = collections.Counter(tok for sent in corpus for tok in sent)
        total  = sum(counts.values())
        self.word_prob = {w: c / total for w, c in counts.items()}
        self.vocab     = set(counts.keys())

    def candidates(self, word: str) -> list:
        """Return candidate corrections within max edit distance."""
        word = word.lower()
        if word in self.vocab:
            return [word]
        cands = []
        for w in self.vocab:
            if abs(len(w) - len(word)) <= self.max_edit_dist:
                d = edit_distance(word, w)
                if d <= self.max_edit_dist:
                    cands.append((w, d))
        return cands

    def correct(self, word: str, top_k: int = 5) -> list:
        """Return top-k corrections ranked by noisy channel score."""
        word = word.lower()
        if word in self.vocab:
            return [(word, 1.0, 0)]

        cands = self.candidates(word)
        if not cands:
            return [(word, 0.0, -1)]

        scored = []
        for w, dist in cands:
            p_word   = self.word_prob.get(w, 1e-9)
            p_channel = math.exp(-dist)          # p(error) decreases with dist
            score    = p_word * p_channel
            scored.append((w, score, dist))

        scored.sort(key=lambda x: -x[1])
        # Normalize scores
        total_score = sum(s for _, s, _ in scored[:top_k]) or 1.0
        return [(w, round(s/total_score, 4), d) for w, s, d in scored[:top_k]]

    def correct_sentence(self, sentence: str) -> str:
        tokens = sentence.lower().split()
        corrected = []
        for tok in tokens:
            if tok.isalpha():
                result = self.correct(tok, top_k=1)
                corrected.append(result[0][0])
            else:
                corrected.append(tok)
        return ' '.join(corrected)


# ═══════════════════════════════════════════════════════════════
# EXERCISE 2 & 3: Word2Vec (CBOW + Skip-gram)
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate_w2v(corpus: list, model_type: str, corpus_name: str,
                            vector_size: int = 100, window: int = 5,
                            min_count: int = 2, epochs: int = 10):
    """Train Word2Vec and evaluate word similarities."""
    sg = 1 if model_type == 'skipgram' else 0
    label = 'Skip-gram' if sg else 'CBOW'
    print(f"\n  Training {label} on {corpus_name}...")

    model = train_word2vec(corpus, vector_size=vector_size, window=window,
                           min_count=min_count, sg=sg, epochs=epochs)
    print(f"    Vocabulary: {len(model.wv)} words")

    # Analogy test: king - man + woman ≈ queen
    analogy_results = {}
    test_pairs = [
        ('king', 'man', 'woman'),
        ('paris', 'france', 'germany'),
    ]
    for pos1, neg1, pos2 in test_pairs:
        try:
            result = model.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=3)
            analogy_results[f"{pos1}-{neg1}+{pos2}"] = [(w, round(s, 4)) for w, s in result]
        except KeyError:
            analogy_results[f"{pos1}-{neg1}+{pos2}"] = []

    # Word similarity
    sim_pairs = [('man', 'woman'), ('king', 'queen'), ('good', 'bad'), ('cat', 'dog')]
    similarities = {}
    for w1, w2 in sim_pairs:
        try:
            sim = model.wv.similarity(w1, w2)
            similarities[f"{w1}-{w2}"] = round(sim, 4)
        except KeyError:
            similarities[f"{w1}-{w2}"] = None

    # Nearest neighbors
    probe_words = ['government', 'people', 'city', 'work']
    neighbors = {}
    for word in probe_words:
        try:
            nn = model.wv.most_similar(word, topn=5)
            neighbors[word] = [(w, round(s, 4)) for w, s in nn]
        except KeyError:
            neighbors[word] = []

    return model, {
        'model_type': label,
        'corpus': corpus_name,
        'vocab_size': len(model.wv),
        'analogies': analogy_results,
        'similarities': similarities,
        'neighbors': neighbors,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_word_vectors_2d(model, model_label: str, corpus_name: str, n_words: int = 80):
    """PCA + t-SNE 2D plots of word vectors."""
    words, vectors = get_word_vectors(model)
    words   = words[:n_words]
    vectors = vectors[:n_words]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, method in zip(axes, ['PCA', 't-SNE']):
        if method == 'PCA':
            coords = reduce_pca(vectors, 2)
        else:
            coords = reduce_tsne(vectors, 2, perplexity=min(20, len(words)-1))

        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=30)
        for i, word in enumerate(words[:40]):
            ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)
        ax.set_title(f'{model_label} — {method} ({corpus_name})')
        ax.set_xlabel(f'{method}-1')
        ax.set_ylabel(f'{method}-2')

    plt.tight_layout()
    fname = f'cat7_{model_label.lower().replace("-","_")}_{corpus_name}.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_similarity_matrix(model, words: list, label: str, corpus_name: str):
    """Cosine similarity matrix for selected words."""
    available = [w for w in words if w in model.wv]
    if len(available) < 2:
        return
    matrix = np.zeros((len(available), len(available)))
    for i, w1 in enumerate(available):
        for j, w2 in enumerate(available):
            try:
                matrix[i, j] = model.wv.similarity(w1, w2)
            except KeyError:
                matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', xticklabels=available,
                yticklabels=available, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(f'{label} Cosine Similarity Matrix ({corpus_name})')
    plt.tight_layout()
    fname = f'cat7_similarity_matrix_{label.lower().replace("-","_")}_{corpus_name}.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_spelling_correction(corrector: SpellingCorrector, test_words: list):
    """Bar chart of correction candidate scores."""
    fig, axes = plt.subplots(1, len(test_words), figsize=(14, 4))
    if len(test_words) == 1:
        axes = [axes]
    for ax, word in zip(axes, test_words):
        results = corrector.correct(word, top_k=6)
        candidates = [r[0] for r in results]
        scores     = [r[1] for r in results]
        ax.barh(candidates[::-1], scores[::-1], color='steelblue')
        ax.set_title(f'Corrections for "{word}"')
        ax.set_xlabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat7_spelling_corrections.png'), dpi=150)
    plt.close()
    print("  Saved: cat7_spelling_corrections.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 7: NEURAL LANGUAGE MODELS")
    print("=" * 60)

    download_data()
    all_metrics = {}

    corpus_brown   = load_corpus('brown',   n=5000)
    corpus_reuters = load_corpus('reuters', n=3000)
    combined       = corpus_brown + corpus_reuters

    # ── Exercise 1: Spelling Correction ─────────────────────────────────────
    print("\n── Exercise 1: Spelling Correction ──")
    corrector = SpellingCorrector(corpus_brown, max_edit_dist=2)

    misspelled = ['goverment', 'recieve', 'occured', 'beutiful', 'definately']
    print(f"\n  {'Misspelled':<15} {'Top Correction':<15} {'Score':>7}")
    print(f"  {'-'*40}")
    spell_results = {}
    for word in misspelled:
        results = corrector.correct(word, top_k=3)
        top = results[0]
        print(f"  {word:<15} {top[0]:<15} {top[1]:>7.4f}")
        spell_results[word] = [(r[0], r[1]) for r in results]

    # Sentence correction
    test_sentences = [
        "the goverment annouced a new polisy",
        "she recieved a beutiful presant",
    ]
    print("\n  Sentence corrections:")
    for sent in test_sentences:
        corrected = corrector.correct_sentence(sent)
        print(f"    Input:    {sent}")
        print(f"    Corrected:{corrected}\n")

    plot_spelling_correction(corrector, misspelled[:4])
    all_metrics['spelling_correction'] = spell_results

    # ── Exercise 2: CBOW ────────────────────────────────────────────────────
    print("\n── Exercise 2: CBOW (Word2Vec) ──")
    cbow_model, cbow_metrics = train_and_evaluate_w2v(
        combined, 'cbow', 'brown+reuters', vector_size=100, epochs=15)

    print(f"  Analogies:")
    for pair, results in cbow_metrics['analogies'].items():
        print(f"    {pair}: {results[:2]}")
    print(f"  Similarities:")
    for pair, sim in cbow_metrics['similarities'].items():
        print(f"    {pair}: {sim}")
    print(f"  Nearest to 'government': {cbow_metrics['neighbors'].get('government', [])[:3]}")

    plot_word_vectors_2d(cbow_model, 'CBOW', 'combined')
    probe_words = ['man', 'woman', 'king', 'queen', 'city', 'country',
                   'good', 'bad', 'work', 'play', 'government', 'people']
    plot_similarity_matrix(cbow_model, probe_words, 'CBOW', 'combined')
    all_metrics['cbow'] = cbow_metrics

    # ── Exercise 3: Skip-gram ────────────────────────────────────────────────
    print("\n── Exercise 3: Skip-gram (Word2Vec) ──")
    sg_model, sg_metrics = train_and_evaluate_w2v(
        combined, 'skipgram', 'brown+reuters', vector_size=100, epochs=15)

    print(f"  Analogies:")
    for pair, results in sg_metrics['analogies'].items():
        print(f"    {pair}: {results[:2]}")
    print(f"  Similarities:")
    for pair, sim in sg_metrics['similarities'].items():
        print(f"    {pair}: {sim}")

    plot_word_vectors_2d(sg_model, 'Skip-gram', 'combined')
    plot_similarity_matrix(sg_model, probe_words, 'Skip-gram', 'combined')
    all_metrics['skipgram'] = sg_metrics

    # Comparison CBOW vs Skip-gram
    print("\n── CBOW vs Skip-gram Comparison ──")
    common_pairs = [(k, v) for k, v in cbow_metrics['similarities'].items() if v is not None]
    print(f"  {'Pair':<20} {'CBOW':>8} {'Skip-gram':>10}")
    print(f"  {'-'*40}")
    for pair, cbow_sim in common_pairs:
        sg_sim = sg_metrics['similarities'].get(pair)
        sg_str = f"{sg_sim:.4f}" if sg_sim is not None else "N/A"
        print(f"  {pair:<20} {cbow_sim:>8.4f} {sg_str:>10}")

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat7_neural_models.json'))
    print("\n  Metrics saved to outputs/metrics/cat7_neural_models.json")
    print("\n[CATEGORY 7 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
