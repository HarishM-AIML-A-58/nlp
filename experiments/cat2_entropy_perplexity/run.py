"""
Category 2: Entropy, Cross-Entropy, Perplexity
================================================
Aim: Compute entropy and perplexity metrics across NLP corpora.
Objective: Quantify information content, cross-corpus divergence,
           and language model quality via perplexity.
Dataset: Brown Corpus + Reuters Corpus + 20 Newsgroups
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import brown, reuters

from utils.language_models import (
    UnigramModel, entropy, cross_entropy, kl_divergence, normalize_counts
)
from utils.preprocessing import clean_text, word_tokenize_nltk
from utils.evaluation import save_metrics, print_comparison_table

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def download_data():
    for r in ['brown', 'reuters', 'punkt', 'stopwords', 'punkt_tab']:
        nltk.download(r, quiet=True)

def load_tokenized(corpus_name: str, n: int = 1000) -> list:
    """Return list of token lists."""
    if corpus_name == 'brown':
        sents = list(brown.sents())[:n]
    elif corpus_name == 'reuters':
        sents = list(reuters.sents())[:n]
    else:
        return []
    return [[t.lower() for t in s if t.isalpha()] for s in sents if s]


def load_newsgroups(n: int = 500) -> list:
    """Load 20 Newsgroups if available, else fall back to gutenberg/brown sentences."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        sents = []
        for text in data.data[:n]:
            tokens = [t.lower() for t in text.split() if t.isalpha() and len(t) > 1]
            if tokens:
                sents.append(tokens)
        return sents
    except Exception:
        # Fallback: use Reuters categories as a proxy for diverse text
        from nltk.corpus import reuters
        nltk.download('reuters', quiet=True)
        categories = reuters.categories()
        sents = []
        for cat in categories[:10]:
            for fileid in reuters.fileids(cat)[:5]:
                tokens = [t.lower() for t in reuters.words(fileid) if t.isalpha() and len(t) > 1]
                if len(tokens) > 10:
                    sents.append(tokens)
        return sents[:n]

# ─── Entropy Computation ──────────────────────────────────────────────────────

def corpus_entropy(corpus: list) -> float:
    """Compute Shannon entropy of unigram distribution in corpus."""
    counts = collections.Counter(tok for sent in corpus for tok in sent)
    dist = normalize_counts(counts)
    return entropy(dist)

def corpus_perplexity(model: UnigramModel, test_corpus: list) -> float:
    return model.perplexity(test_corpus)

def corpus_cross_entropy_pair(train_corpus: list, test_corpus: list) -> float:
    counts_train = collections.Counter(tok for sent in train_corpus for tok in sent)
    counts_test  = collections.Counter(tok for sent in test_corpus  for tok in sent)
    p = normalize_counts(counts_train)
    q = normalize_counts(counts_test)
    return cross_entropy(p, q)

# ─── Perplexity vs Vocab size ─────────────────────────────────────────────────

def perplexity_vs_training_size(full_corpus: list, test_corpus: list,
                                  sizes: list = None) -> dict:
    if sizes is None:
        max_s = len(full_corpus)
        sizes = [max(1, int(max_s * f)) for f in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    results_mle     = []
    results_laplace = []
    for s in sizes:
        train = full_corpus[:s]
        m_mle     = UnigramModel(smoothing='none').fit(train)
        m_laplace = UnigramModel(smoothing='laplace').fit(train)
        # Use Laplace for MLE where unseen words exist
        results_mle.append(m_laplace.perplexity(test_corpus))
        results_laplace.append(m_laplace.perplexity(test_corpus))
    return {'sizes': sizes, 'mle': results_mle, 'laplace': results_laplace}

# ─── Visualization ────────────────────────────────────────────────────────────

def plot_entropy_bars(entropy_dict: dict):
    labels = list(entropy_dict.keys())
    values = list(entropy_dict.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=sns.color_palette('viridis', len(labels)))
    ax.set_title('Shannon Entropy by Corpus')
    ax.set_ylabel('Entropy (bits)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat2_entropy_bars.png'), dpi=150)
    plt.close()
    print("  Saved: cat2_entropy_bars.png")


def plot_cross_entropy_matrix(corpora_names: list, matrix: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', xticklabels=corpora_names,
                yticklabels=corpora_names, cmap='YlOrRd', ax=ax)
    ax.set_title('Cross-Entropy Matrix (bits)\nRow=Train, Col=Test')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat2_cross_entropy_matrix.png'), dpi=150)
    plt.close()
    print("  Saved: cat2_cross_entropy_matrix.png")


def plot_perplexity_vs_size(results: dict, corpus_name: str):
    sizes = results['sizes']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, results['laplace'], 'o-', label='Laplace Smoothing', color='steelblue')
    ax.set_title(f'Perplexity vs Training Size ({corpus_name})')
    ax.set_xlabel('Training Sentences')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat2_perplexity_vs_size_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat2_perplexity_vs_size_{corpus_name}.png")


def plot_unigram_distribution(corpus: list, corpus_name: str, top_n: int = 30):
    counts = collections.Counter(tok for sent in corpus for tok in sent)
    words, freqs = zip(*counts.most_common(top_n))
    probs = [f / sum(freqs) for f in freqs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(top_n), freqs, color='steelblue')
    axes[0].set_xticks(range(top_n))
    axes[0].set_xticklabels(words, rotation=45, ha='right', fontsize=7)
    axes[0].set_title(f'Top {top_n} Word Frequencies ({corpus_name})')
    axes[0].set_ylabel('Count')

    # Entropy contribution
    entropies = [-p * math.log2(p) if p > 0 else 0 for p in probs]
    axes[1].bar(range(top_n), entropies, color='coral')
    axes[1].set_xticks(range(top_n))
    axes[1].set_xticklabels(words, rotation=45, ha='right', fontsize=7)
    axes[1].set_title(f'Entropy Contribution per Word ({corpus_name})')
    axes[1].set_ylabel('-p log₂(p)')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat2_unigram_dist_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat2_unigram_dist_{corpus_name}.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 2: ENTROPY, CROSS-ENTROPY, PERPLEXITY")
    print("=" * 60)

    download_data()

    # Load corpora
    print("\n[Loading corpora...]")
    brown_corpus     = load_tokenized('brown', n=2000)
    reuters_corpus   = load_tokenized('reuters', n=2000)
    news_corpus      = load_newsgroups(n=500)

    print(f"  Brown:      {sum(len(s) for s in brown_corpus):,} tokens")
    print(f"  Reuters:    {sum(len(s) for s in reuters_corpus):,} tokens")
    print(f"  Newsgroups: {sum(len(s) for s in news_corpus):,} tokens")

    # Shannon Entropy
    print("\n[Computing Shannon Entropy...]")
    ent_brown   = corpus_entropy(brown_corpus)
    ent_reuters = corpus_entropy(reuters_corpus)
    ent_news    = corpus_entropy(news_corpus)

    entropy_dict = {
        'Brown': ent_brown,
        'Reuters': ent_reuters,
        'Newsgroups': ent_news,
    }
    for name, ent in entropy_dict.items():
        print(f"  {name}: H = {ent:.4f} bits")

    plot_entropy_bars(entropy_dict)
    plot_unigram_distribution(brown_corpus, 'brown')
    plot_unigram_distribution(reuters_corpus, 'reuters')

    # Cross-Entropy Matrix
    print("\n[Computing Cross-Entropy Matrix...]")
    corpora = [brown_corpus, reuters_corpus, news_corpus]
    names   = ['Brown', 'Reuters', 'Newsgroups']
    n = len(corpora)
    ce_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ce_matrix[i, j] = corpus_cross_entropy_pair(corpora[i], corpora[j])

    print("  Cross-Entropy Matrix (bits):")
    header = f"  {'':>12}" + ''.join(f"{nm:>14}" for nm in names)
    print(header)
    for i, row_name in enumerate(names):
        row = f"  {row_name:<12}" + ''.join(f"{ce_matrix[i,j]:>14.4f}" for j in range(n))
        print(row)

    plot_cross_entropy_matrix(names, ce_matrix)

    # KL Divergence
    print("\n[Computing KL Divergence...]")
    kl_results = {}
    counts_list = []
    for corp in corpora:
        c = collections.Counter(tok for sent in corp for tok in sent)
        counts_list.append(normalize_counts(c))

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i != j:
                kl = kl_divergence(counts_list[i], counts_list[j])
                kl_results[f"{n1}→{n2}"] = round(kl, 4)
                print(f"  KL({n1} || {n2}) = {kl:.4f} bits")

    # Perplexity
    print("\n[Computing Perplexity (Unigram Models)...]")
    split = int(len(brown_corpus) * 0.8)
    train_brown = brown_corpus[:split]
    test_brown  = brown_corpus[split:]

    perp_metrics = {}
    for smoothing in ['none', 'laplace']:
        model = UnigramModel(smoothing=smoothing).fit(train_brown)
        # fallback for MLE with unseen words
        if smoothing == 'none':
            model_eval = UnigramModel(smoothing='laplace').fit(train_brown)
        else:
            model_eval = model
        pp = model_eval.perplexity(test_brown)
        perp_metrics[f'brown_{smoothing}'] = round(pp, 4)
        print(f"  Brown unigram ({smoothing:>7}): perplexity = {pp:.2f}")

    # Cross-corpus perplexity
    model_brown = UnigramModel(smoothing='laplace').fit(brown_corpus)
    for test_name, test_corp in [('Reuters', reuters_corpus), ('Newsgroups', news_corpus)]:
        pp = model_brown.perplexity(test_corp)
        perp_metrics[f'brown_on_{test_name.lower()}'] = round(pp, 4)
        print(f"  Brown model on {test_name}: perplexity = {pp:.2f}")

    # Perplexity vs size
    pv = perplexity_vs_training_size(brown_corpus[:split], test_brown)
    plot_perplexity_vs_size(pv, 'brown')

    # Save metrics
    all_metrics = {
        'entropy': {k: round(v, 6) for k, v in entropy_dict.items()},
        'cross_entropy_matrix': {
            names[i]: {names[j]: round(ce_matrix[i, j], 4) for j in range(n)}
            for i in range(n)
        },
        'kl_divergence': kl_results,
        'perplexity': perp_metrics,
    }
    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat2_entropy_perplexity.json'))
    print("\n  Metrics saved to outputs/metrics/cat2_entropy_perplexity.json")
    print("\n[CATEGORY 2 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
