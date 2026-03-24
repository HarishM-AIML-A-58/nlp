"""
Category 6: Bigram Language Model
===================================
Aim: Build and evaluate a bigram language model with multiple smoothing techniques.
Objective: Implement MLE, Laplace, Good-Turing, and Kneser-Ney smoothing;
           evaluate with perplexity; generate text from the model.
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

from utils.language_models import BigramModel, UnigramModel
from utils.evaluation import save_metrics, print_comparison_table

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)


def download_data():
    for r in ['brown', 'reuters', 'punkt', 'punkt_tab']:
        nltk.download(r, quiet=True)


def load_corpus(name: str, n: int = 3000) -> list:
    if name == 'brown':
        return [[t.lower() for t in s if t.isalpha()] for s in list(brown.sents())[:n]]
    elif name == 'reuters':
        return [[t.lower() for t in s if t.isalpha()] for s in list(reuters.sents())[:n]]
    return []


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_bigram_heatmap(model: BigramModel, top_n: int = 15, corpus_name: str = ''):
    """Plot heatmap of bigram transition probabilities for top-N words."""
    top_words = [w for w, _ in
                 sorted(model.unigram_counts.items(), key=lambda x: -x[1])
                 if w not in (BigramModel.BOS, BigramModel.EOS)][:top_n]

    matrix = np.zeros((top_n, top_n))
    for i, w1 in enumerate(top_words):
        for j, w2 in enumerate(top_words):
            matrix[i, j] = model.prob(w2, w1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, xticklabels=top_words, yticklabels=top_words,
                cmap='Blues', ax=ax, fmt='.3f', annot=True,
                annot_kws={'size': 7})
    ax.set_title(f'Bigram Transition Probabilities — Top {top_n} words ({corpus_name})')
    ax.set_xlabel('Next Word')
    ax.set_ylabel('Current Word')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat6_bigram_heatmap_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat6_bigram_heatmap_{corpus_name}.png")


def plot_perplexity_comparison(results: dict, corpus_name: str):
    methods = list(results.keys())
    values  = list(results.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette('Set2', len(methods))
    bars = ax.bar(methods, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10)
    ax.set_title(f'Bigram Model Perplexity by Smoothing ({corpus_name})')
    ax.set_ylabel('Perplexity (lower = better)')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat6_perplexity_comparison_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat6_perplexity_comparison_{corpus_name}.png")


def plot_generated_text_lengths(generated: list, corpus_name: str):
    lengths = [len(g) for g in generated]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=15, color='steelblue', edgecolor='white')
    ax.set_title(f'Generated Sentence Lengths ({corpus_name})')
    ax.set_xlabel('Words')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat6_generated_lengths_{corpus_name}.png'), dpi=150)
    plt.close()


def plot_top_bigrams(model: BigramModel, top_n: int = 20, corpus_name: str = ''):
    items = [(bg, cnt) for bg, cnt in
             sorted(model.bigram_counts.items(), key=lambda x: -x[1])
             if BigramModel.BOS not in bg and BigramModel.EOS not in bg][:top_n]
    labels = [f"{w1} → {w2}" for (w1, w2), _ in items]
    counts = [cnt for _, cnt in items]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], counts[::-1], color='coral')
    ax.set_title(f'Top {top_n} Bigrams ({corpus_name})')
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat6_top_bigrams_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat6_top_bigrams_{corpus_name}.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 6: BIGRAM LANGUAGE MODEL")
    print("=" * 60)

    download_data()
    all_metrics = {}

    smoothing_methods = ['laplace', 'good_turing', 'kneser_ney']

    for corpus_name in ['brown', 'reuters']:
        print(f"\n[{corpus_name.upper()} CORPUS]")
        corpus = load_corpus(corpus_name, n=3000)
        split  = int(len(corpus) * 0.8)
        train, test = corpus[:split], corpus[split:]

        print(f"  Train: {split} sents | Test: {len(test)} sents")

        models = {}
        perplexity_results = {}

        for method in smoothing_methods:
            model = BigramModel(smoothing=method).fit(train)
            models[method] = model
            pp = model.perplexity(test)
            perplexity_results[method] = round(pp, 2)
            print(f"  Perplexity ({method:<12}): {pp:.2f}")

        # Vocab and bigram stats
        best_model = models['kneser_ney']
        print(f"\n  Unigram vocab size: {len(best_model.vocab):,}")
        print(f"  Unique bigrams:     {len(best_model.bigram_counts):,}")

        # Top bigrams
        top_bgs = sorted(
            [(bg, cnt) for bg, cnt in best_model.bigram_counts.items()
             if BigramModel.BOS not in bg and BigramModel.EOS not in bg],
            key=lambda x: -x[1]
        )[:10]
        print(f"\n  Top 10 Bigrams:")
        for (w1, w2), cnt in top_bgs:
            p = best_model.prob(w2, w1)
            print(f"    {w1:<12} → {w2:<12}  count={cnt:>5}  P={p:.5f}")

        # Sample probabilities
        test_bigrams = [('the', 'government'), ('in', 'the'), ('xyz', 'abc')]
        print(f"\n  Sample Bigram Probabilities:")
        for w1, w2 in test_bigrams:
            for method, model in models.items():
                p = model.prob(w2, w1)
                print(f"    P({w2}|{w1}) [{method}]: {p:.6f}")
            print()

        # Text generation
        print(f"  Generated sentences (Kneser-Ney):")
        generated_sents = []
        for seed in ['the', 'government', 'in']:
            gen = models['kneser_ney'].generate(seed=seed, max_len=15)
            generated_sents.append(gen)
            print(f"    [{seed}] → {' '.join(gen)}")

        # Plots
        plot_bigram_heatmap(models['laplace'], top_n=12, corpus_name=corpus_name)
        plot_perplexity_comparison(perplexity_results, corpus_name)
        plot_top_bigrams(best_model, top_n=20, corpus_name=corpus_name)
        plot_generated_text_lengths(generated_sents, corpus_name)

        all_metrics[corpus_name] = {
            'vocab_size': len(best_model.vocab),
            'unique_bigrams': len(best_model.bigram_counts),
            'perplexity': perplexity_results,
        }

    # Cross-corpus evaluation
    print("\n[Cross-Corpus Evaluation]")
    brown_corpus   = load_corpus('brown',   n=2000)
    reuters_corpus = load_corpus('reuters', n=2000)
    model_brown   = BigramModel(smoothing='kneser_ney').fit(brown_corpus[:1600])
    model_reuters = BigramModel(smoothing='kneser_ney').fit(reuters_corpus[:1600])

    cross_results = {
        'Brown→Brown':     round(model_brown.perplexity(brown_corpus[1600:]), 2),
        'Brown→Reuters':   round(model_brown.perplexity(reuters_corpus[1600:]), 2),
        'Reuters→Reuters': round(model_reuters.perplexity(reuters_corpus[1600:]), 2),
        'Reuters→Brown':   round(model_reuters.perplexity(brown_corpus[1600:]), 2),
    }
    for pair, pp in cross_results.items():
        print(f"  {pair:<25}: {pp:.2f}")

    all_metrics['cross_corpus'] = cross_results

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat6_bigram_model.json'))
    print("\n  Metrics saved to outputs/metrics/cat6_bigram_model.json")
    print("\n[CATEGORY 6 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
