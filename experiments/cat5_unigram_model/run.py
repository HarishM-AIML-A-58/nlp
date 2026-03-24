"""
Category 5: Unigram Language Model
=====================================
Aim: Build and evaluate a unigram language model on NLP corpora.
Objective: Estimate word probabilities, apply Laplace smoothing,
           compare MLE vs smoothed distributions, compute perplexity.
Dataset: Brown Corpus + Reuters Corpus
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

from utils.language_models import UnigramModel, normalize_counts, entropy
from utils.evaluation import save_metrics, print_comparison_table

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# ─── Data ─────────────────────────────────────────────────────────────────────

def download_data():
    for r in ['brown', 'reuters', 'punkt', 'punkt_tab']:
        nltk.download(r, quiet=True)

def load_corpus(name: str, n: int = 3000) -> list:
    if name == 'brown':
        return [[t.lower() for t in s if t.isalpha()] for s in list(brown.sents())[:n]]
    elif name == 'reuters':
        return [[t.lower() for t in s if t.isalpha()] for s in list(reuters.sents())[:n]]
    return []

# ─── Zipf's Law Analysis ──────────────────────────────────────────────────────

def plot_zipf(model: UnigramModel, corpus_name: str):
    """Plot rank-frequency (Zipf's Law) distribution."""
    counts = sorted(model.counts.values(), reverse=True)
    ranks  = range(1, len(counts) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    axes[0].plot(list(ranks)[:200], counts[:200], color='steelblue')
    axes[0].set_title(f"Rank-Frequency (linear) — {corpus_name}")
    axes[0].set_xlabel('Rank')
    axes[0].set_ylabel('Frequency')

    # Log-log scale (Zipf's law)
    log_ranks  = [math.log10(r) for r in list(ranks)[:1000]]
    log_counts = [math.log10(max(c, 1)) for c in counts[:1000]]
    axes[1].plot(log_ranks, log_counts, 'o', markersize=2, color='coral')
    axes[1].set_title(f"Zipf's Law (log-log) — {corpus_name}")
    axes[1].set_xlabel('log₁₀(Rank)')
    axes[1].set_ylabel('log₁₀(Frequency)')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat5_zipf_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat5_zipf_{corpus_name}.png")


def plot_probability_comparison(model_mle: UnigramModel, model_lp: UnigramModel,
                                 corpus_name: str, top_n: int = 20):
    """Compare MLE vs Laplace probability for top words."""
    top_words = [w for w, _ in model_mle.top_n(top_n)]
    probs_mle = [model_mle.prob(w)  for w in top_words]
    probs_lp  = [model_lp.prob(w)   for w in top_words]

    x = np.arange(top_n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, probs_mle, width, label='MLE',    color='steelblue')
    ax.bar(x + width/2, probs_lp,  width, label='Laplace', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(top_words, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'MLE vs Laplace Probabilities — Top {top_n} Words ({corpus_name})')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat5_prob_comparison_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat5_prob_comparison_{corpus_name}.png")


def plot_perplexity_smoothing(results: dict, corpus_name: str):
    labels = list(results.keys())
    values = list(results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=sns.color_palette('Set2', len(labels)))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10)
    ax.set_title(f'Perplexity by Smoothing Method ({corpus_name})')
    ax.set_ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat5_perplexity_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat5_perplexity_{corpus_name}.png")


def plot_word_probability_distribution(model: UnigramModel, corpus_name: str):
    """Histogram of log-probabilities."""
    log_probs = [math.log10(max(model.prob(w), 1e-12)) for w in model.counts]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(log_probs, bins=40, color='steelblue', edgecolor='white')
    ax.set_title(f'Distribution of log₁₀(P(word)) — {corpus_name}')
    ax.set_xlabel('log₁₀(Probability)')
    ax.set_ylabel('Number of Words')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat5_log_prob_dist_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat5_log_prob_dist_{corpus_name}.png")

# ─── Text Generation ──────────────────────────────────────────────────────────

def generate_unigram(model: UnigramModel, n: int = 20) -> str:
    """Sample n words from unigram distribution."""
    import random
    words = list(model.counts.keys())
    probs = [model.prob(w) for w in words]
    total = sum(probs)
    probs = [p / total for p in probs]
    return ' '.join(random.choices(words, probs, k=n))

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 5: UNIGRAM LANGUAGE MODEL")
    print("=" * 60)

    download_data()
    all_metrics = {}

    for corpus_name in ['brown', 'reuters']:
        print(f"\n[{corpus_name.upper()} CORPUS]")
        corpus = load_corpus(corpus_name, n=3000)

        # Train-test split
        split = int(len(corpus) * 0.8)
        train, test = corpus[:split], corpus[split:]

        # Train models
        model_mle    = UnigramModel(smoothing='none').fit(train)
        model_laplace= UnigramModel(smoothing='laplace', k=1.0).fit(train)
        model_half   = UnigramModel(smoothing='laplace', k=0.5).fit(train)

        print(f"  Training sentences: {split}")
        print(f"  Vocabulary size:    {model_mle.vocab_size:,}")
        print(f"  Total tokens:       {model_mle.total:,}")

        # Top words
        print(f"\n  Top 10 words (MLE):")
        for word, prob in model_mle.top_n(10):
            print(f"    {word:<15} P={prob:.6f}  log P={math.log(prob):.4f}")

        # Perplexity
        pp_laplace = model_laplace.perplexity(test)
        pp_half    = model_half.perplexity(test)

        # For MLE, use Laplace to handle unseen words
        pp_mle_safe = model_laplace.perplexity(test)

        perplexity_results = {
            'MLE (Laplace k=1)': round(pp_laplace, 2),
            'Laplace k=0.5':     round(pp_half, 2),
        }
        print(f"\n  Perplexity Comparison:")
        for method, pp in perplexity_results.items():
            print(f"    {method:<22}: {pp:.2f}")

        # Entropy
        dist = normalize_counts(model_mle.counts)
        ent  = entropy(dist)
        print(f"  Corpus entropy: {ent:.4f} bits")

        # Probability of specific words/sentences
        test_words = ['the', 'government', 'economy', 'xyz_unknown']
        print(f"\n  Sample word probabilities:")
        for w in test_words:
            p_mle    = model_mle.prob(w)
            p_laplace= model_laplace.prob(w)
            print(f"    {w:<15} MLE={p_mle:.6f}  Laplace={p_laplace:.6f}")

        # Text generation
        sample_text = generate_unigram(model_laplace, n=15)
        print(f"\n  Generated text: {sample_text}")

        # Visualize
        plot_zipf(model_mle, corpus_name)
        plot_probability_comparison(model_mle, model_laplace, corpus_name)
        plot_perplexity_smoothing(perplexity_results, corpus_name)
        plot_word_probability_distribution(model_laplace, corpus_name)

        all_metrics[corpus_name] = {
            'vocab_size': model_mle.vocab_size,
            'total_tokens': model_mle.total,
            'entropy_bits': round(ent, 4),
            'perplexity': perplexity_results,
        }

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat5_unigram_model.json'))
    print("\n  Metrics saved to outputs/metrics/cat5_unigram_model.json")
    print("\n[CATEGORY 5 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
