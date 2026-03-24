"""
Category 1: Text Preprocessing
================================
Aim: Demonstrate text preprocessing pipeline on NLP corpora.
Objective: Apply normalization, cleaning, tokenization, stopword removal,
           stemming, and lemmatization; compare before vs. after results.
Dataset: Brown Corpus (NLTK) + Reuters Corpus (NLTK)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import brown, reuters
from utils.preprocessing import (
    clean_text, word_tokenize_nltk, remove_stopwords,
    stem_porter, stem_snowball, lemmatize_nltk, full_pipeline
)
from utils.evaluation import save_metrics, save_csv

# ─── Setup ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR= os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# ─── Download Data ────────────────────────────────────────────────────────────

def download_data():
    for resource in ['brown', 'reuters', 'punkt', 'stopwords', 'wordnet',
                     'averaged_perceptron_tagger', 'punkt_tab', 'omw-1.4']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# ─── Load Corpus ──────────────────────────────────────────────────────────────

def load_corpus(corpus_name: str = 'brown', n_sentences: int = 500) -> list:
    if corpus_name == 'brown':
        sents = brown.sents()[:n_sentences]
        return [' '.join(s) for s in sents]
    elif corpus_name == 'reuters':
        sents = reuters.sents()[:n_sentences]
        return [' '.join(s) for s in sents]
    return []

# ─── Analysis Helpers ─────────────────────────────────────────────────────────

def token_statistics(texts: list) -> dict:
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.split())
    freq = collections.Counter(all_tokens)
    return {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(freq),
        'top_10': freq.most_common(10),
        'avg_token_length': sum(len(t) for t in all_tokens) / max(len(all_tokens), 1),
    }

def compare_stages(raw_texts: list) -> dict:
    stages = {'raw': [], 'cleaned': [], 'no_stops': [], 'stemmed': [], 'lemmatized': []}
    for text in raw_texts[:100]:
        result = full_pipeline(text, stemming=True, lemmatization=True)
        stages['raw'].append(text)
        stages['cleaned'].append(result['cleaned'])
        stages['no_stops'].append(' '.join(result['tokens_no_stopwords']))
        stages['stemmed'].append(' '.join(result['stemmed'] or []))
        stages['lemmatized'].append(' '.join(result['lemmatized'] or []))
    return stages

# ─── Visualization ────────────────────────────────────────────────────────────

def plot_token_reduction(stats_by_stage: dict, corpus_name: str):
    stages = list(stats_by_stage.keys())
    token_counts = [stats_by_stage[s]['total_tokens'] for s in stages]
    vocab_counts  = [stats_by_stage[s]['unique_tokens'] for s in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(stages, token_counts, color=sns.color_palette('Blues_d', len(stages)))
    ax1.set_title(f'Total Token Count by Stage ({corpus_name})')
    ax1.set_ylabel('Token Count')
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(stages, vocab_counts, color=sns.color_palette('Greens_d', len(stages)))
    ax2.set_title(f'Vocabulary Size by Stage ({corpus_name})')
    ax2.set_ylabel('Vocabulary Size')
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat1_token_reduction_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat1_token_reduction_{corpus_name}.png")


def plot_top_words(stats: dict, stage: str, corpus_name: str, top_n: int = 15):
    words = [w for w, _ in stats['top_10'][:top_n]]
    counts = [c for _, c in stats['top_10'][:top_n]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(words[::-1], counts[::-1], color='steelblue')
    ax.set_title(f'Top {top_n} Words — {stage} ({corpus_name})')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat1_top_words_{stage}_{corpus_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: cat1_top_words_{stage}_{corpus_name}.png")


def plot_word_length_distribution(texts: list, stage: str, corpus_name: str):
    all_tokens = [t for text in texts for t in text.split()]
    lengths = [len(t) for t in all_tokens if t.isalpha()]
    length_counts = collections.Counter(lengths)
    x = sorted(length_counts.keys())
    y = [length_counts[k] for k in x]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color='coral')
    ax.set_title(f'Word Length Distribution — {stage} ({corpus_name})')
    ax.set_xlabel('Word Length')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cat1_word_length_{stage}_{corpus_name}.png'), dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 1: TEXT PREPROCESSING")
    print("=" * 60)

    download_data()

    all_metrics = {}

    for corpus_name in ['brown', 'reuters']:
        print(f"\n[{corpus_name.upper()} CORPUS]")
        raw_texts = load_corpus(corpus_name, n_sentences=500)
        print(f"  Loaded {len(raw_texts)} sentences")

        stages_data = compare_stages(raw_texts)

        # Compute stats for each stage
        stats_by_stage = {}
        for stage, texts in stages_data.items():
            stats_by_stage[stage] = token_statistics(texts)

        # Print comparison
        print(f"\n  {'Stage':<15} {'Total Tokens':>14} {'Vocab Size':>12}")
        print(f"  {'-'*42}")
        for stage, stats in stats_by_stage.items():
            print(f"  {stage:<15} {stats['total_tokens']:>14,} {stats['unique_tokens']:>12,}")

        # Show example
        example = full_pipeline(raw_texts[0])
        print(f"\n  Example transformation:")
        print(f"    Original   : {example['original'][:100]}...")
        print(f"    Cleaned    : {example['cleaned'][:100]}...")
        print(f"    Tokens     : {example['tokens'][:10]}")
        print(f"    No stops   : {example['tokens_no_stopwords'][:10]}")
        print(f"    Stemmed    : {(example['stemmed'] or [])[:10]}")
        print(f"    Lemmatized : {(example['lemmatized'] or [])[:10]}")

        # Visualize
        plot_token_reduction(stats_by_stage, corpus_name)
        plot_top_words(stats_by_stage['raw'], 'raw', corpus_name)
        plot_top_words(stats_by_stage['lemmatized'], 'lemmatized', corpus_name)
        plot_word_length_distribution(stages_data['lemmatized'], 'lemmatized', corpus_name)

        all_metrics[corpus_name] = {
            stage: {
                'total_tokens': s['total_tokens'],
                'unique_tokens': s['unique_tokens'],
                'avg_token_length': round(s['avg_token_length'], 3),
            }
            for stage, s in stats_by_stage.items()
        }

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat1_preprocessing.json'))
    print(f"\n  Metrics saved to outputs/metrics/cat1_preprocessing.json")
    print("\n[CATEGORY 1 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
