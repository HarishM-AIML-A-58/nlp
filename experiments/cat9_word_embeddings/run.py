"""
Category 9: Word Embeddings + Visualization
=============================================
Aim: Train word embeddings and visualize semantic structure using
     dimensionality reduction techniques.
Objective: Train Word2Vec; reduce to 2D via PCA, t-SNE, UMAP;
           visualize word clusters and analogies.
Dataset: Brown Corpus + Reuters Corpus (combined)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import brown, reuters

from utils.embeddings import (
    train_word2vec, get_word_vectors,
    reduce_pca, reduce_tsne, reduce_umap,
    most_similar
)
from utils.evaluation import save_metrics

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# Semantic categories for visualization
WORD_CATEGORIES = {
    'Countries':  ['france', 'germany', 'england', 'japan', 'china', 'india', 'russia', 'italy'],
    'Cities':     ['paris', 'berlin', 'london', 'tokyo', 'rome', 'madrid', 'moscow', 'beijing'],
    'Animals':    ['cat', 'dog', 'horse', 'bird', 'fish', 'lion', 'tiger', 'bear'],
    'Colors':     ['red', 'blue', 'green', 'black', 'white', 'yellow', 'brown', 'orange'],
    'Numbers':    ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
    'Royalty':    ['king', 'queen', 'prince', 'princess', 'lord', 'duke', 'emperor', 'empress'],
    'Verbs':      ['run', 'walk', 'eat', 'sleep', 'work', 'play', 'read', 'write'],
    'Adjectives': ['good', 'bad', 'big', 'small', 'fast', 'slow', 'strong', 'weak'],
}


def download_data():
    for r in ['brown', 'reuters', 'punkt', 'punkt_tab']:
        nltk.download(r, quiet=True)


def load_corpus(n_brown: int = 8000, n_reuters: int = 5000) -> list:
    brown_sents   = [[t.lower() for t in s if t.isalpha() and len(t) > 1]
                     for s in list(brown.sents())[:n_brown] if len(s) > 3]
    reuters_sents = [[t.lower() for t in s if t.isalpha() and len(t) > 1]
                     for s in list(reuters.sents())[:n_reuters] if len(s) > 3]
    return brown_sents + reuters_sents


# ─── Embedding Training ───────────────────────────────────────────────────────

def train_models(corpus: list) -> dict:
    """Train CBOW and Skip-gram models at multiple vector sizes."""
    models = {}
    configs = [
        ('cbow_100',    0, 100),
        ('skipgram_100',1, 100),
        ('cbow_50',     0,  50),
    ]
    for name, sg, size in configs:
        print(f"  Training {name} (size={size})...")
        model = train_word2vec(corpus, vector_size=size, window=5,
                               min_count=3, sg=sg, epochs=15)
        models[name] = model
        print(f"    Vocab: {len(model.wv):,} words")
    return models


# ─── Visualization Functions ──────────────────────────────────────────────────

def plot_embeddings_2d(vectors_2d: np.ndarray, words: list, categories: dict,
                        method: str, model_name: str):
    """Scatter plot with color-coded semantic categories."""
    # Map words to categories
    word_to_cat = {}
    for cat, cat_words in categories.items():
        for w in cat_words:
            word_to_cat[w] = cat

    cat_list   = list(categories.keys())
    cmap       = plt.colormaps.get_cmap('tab10').resampled(len(cat_list))
    cat_colors = {cat: cmap(i) for i, cat in enumerate(cat_list)}

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot by category
    for cat in cat_list:
        cat_idx = [i for i, w in enumerate(words) if word_to_cat.get(w) == cat]
        if not cat_idx:
            continue
        xs = vectors_2d[cat_idx, 0]
        ys = vectors_2d[cat_idx, 1]
        ax.scatter(xs, ys, c=[cat_colors[cat]], label=cat, s=80, alpha=0.8)
        for idx in cat_idx:
            ax.annotate(words[idx], (vectors_2d[idx, 0], vectors_2d[idx, 1]),
                        fontsize=9, alpha=0.9,
                        xytext=(3, 3), textcoords='offset points')

    ax.set_title(f'{model_name} — {method} 2D Embedding Visualization', fontsize=13)
    ax.set_xlabel(f'{method} Dimension 1')
    ax.set_ylabel(f'{method} Dimension 2')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f'cat9_{model_name.lower()}_{method.lower().replace("-","_")}.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_analogy_vectors(model, analogies: list, model_name: str):
    """Plot word analogy parallelograms using PCA."""
    valid_analogies = []
    all_words_needed = set()
    for a, b, c, d in analogies:
        if all(w in model.wv for w in [a, b, c, d]):
            valid_analogies.append((a, b, c, d))
            all_words_needed.update([a, b, c, d])

    if not valid_analogies:
        return

    all_words = list(all_words_needed)
    all_vecs  = np.array([model.wv[w] for w in all_words])
    coords_2d = reduce_pca(all_vecs, 2)
    w2coord   = {w: coords_2d[i] for i, w in enumerate(all_words)}

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette('Set1', len(valid_analogies))

    for (a, b, c, d), color in zip(valid_analogies, colors):
        pts = np.array([w2coord[a], w2coord[b], w2coord[d], w2coord[c], w2coord[a]])
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=color,
                label=f'{a}:{b}::{c}:{d}', alpha=0.7)
        for w in [a, b, c, d]:
            x, y = w2coord[w]
            ax.annotate(w, (x, y), fontsize=9,
                        xytext=(4, 4), textcoords='offset points')

    ax.set_title(f'{model_name} — Word Analogy Parallelograms (PCA)', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'cat9_{model_name.lower()}_analogies.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_similarity_heatmap(model, word_groups: dict, model_name: str):
    """Similarity heatmap for words within semantic categories."""
    all_words = []
    group_labels = []
    for cat, words in word_groups.items():
        available = [w for w in words if w in model.wv][:6]
        all_words.extend(available)
        group_labels.extend([cat] * len(available))

    if len(all_words) < 2:
        return

    n = len(all_words)
    sim_matrix = np.zeros((n, n))
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            sim_matrix[i, j] = model.wv.similarity(w1, w2)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sim_matrix, xticklabels=all_words, yticklabels=all_words,
                cmap='RdYlGn', center=0, vmin=-0.3, vmax=1.0,
                ax=ax, annot=True, fmt='.2f', annot_kws={'size': 7})
    ax.set_title(f'{model_name} — Word Similarity Heatmap', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fname = f'cat9_{model_name.lower()}_similarity_heatmap.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_reduction_comparison(words_2d_dict: dict, words: list, title: str, fname: str):
    """Side-by-side comparison of PCA, t-SNE, UMAP."""
    methods = list(words_2d_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        coords = words_2d_dict[method]
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=25, color='steelblue')
        for i, w in enumerate(words[:50]):
            ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)
        ax.set_title(method)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_analogies(model, test_analogies: list) -> dict:
    """Evaluate word analogy accuracy."""
    correct = 0
    total   = 0
    results = []
    for a, b, c, expected_d in test_analogies:
        try:
            predicted = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
            predicted_word = predicted[0][0] if predicted else ''
            is_correct     = predicted_word == expected_d
            correct += int(is_correct)
            total   += 1
            results.append({
                'analogy': f'{a}:{b}::{c}:{expected_d}',
                'predicted': predicted_word,
                'correct': is_correct,
            })
        except KeyError:
            pass
    accuracy = correct / max(total, 1)
    return {'accuracy': round(accuracy, 4), 'correct': correct, 'total': total, 'results': results}


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 9: WORD EMBEDDINGS + VISUALIZATION")
    print("=" * 60)

    download_data()

    # Load corpus
    print("\n[Loading corpus...]")
    corpus = load_corpus(n_brown=8000, n_reuters=5000)
    total_tokens = sum(len(s) for s in corpus)
    print(f"  Sentences: {len(corpus):,}")
    print(f"  Tokens:    {total_tokens:,}")

    # Train models
    print("\n[Training Word2Vec models...]")
    models = train_models(corpus)

    all_metrics = {}

    # ── CBOW-100: Full Analysis ──────────────────────────────────────────────
    main_model = models['cbow_100']
    model_name = 'CBOW_100'
    words, vectors = get_word_vectors(main_model)

    # Filter to words in our category list
    category_words = [w for cat_words in WORD_CATEGORIES.values()
                      for w in cat_words if w in main_model.wv]
    category_vecs  = np.array([main_model.wv[w] for w in category_words])

    print(f"\n[Dimensionality Reduction: {len(category_words)} category words]")

    # PCA
    print("  Computing PCA...")
    pca_2d = reduce_pca(category_vecs, 2)
    plot_embeddings_2d(pca_2d, category_words, WORD_CATEGORIES, 'PCA', model_name)

    # t-SNE
    print("  Computing t-SNE...")
    tsne_2d = reduce_tsne(category_vecs, 2, perplexity=min(15, len(category_words)-1))
    plot_embeddings_2d(tsne_2d, category_words, WORD_CATEGORIES, 't-SNE', model_name)

    # UMAP
    print("  Computing UMAP...")
    umap_2d = reduce_umap(category_vecs, 2, n_neighbors=min(10, len(category_words)-1))
    plot_embeddings_2d(umap_2d, category_words, WORD_CATEGORIES, 'UMAP', model_name)

    # Reduction comparison on larger set
    n_words_large = min(200, len(words))
    large_vecs  = vectors[:n_words_large]
    large_words = words[:n_words_large]

    reductions = {
        'PCA':    reduce_pca(large_vecs, 2),
        't-SNE':  reduce_tsne(large_vecs, 2, perplexity=min(30, n_words_large-1)),
        'UMAP':   reduce_umap(large_vecs, 2, n_neighbors=min(15, n_words_large-1)),
    }
    plot_reduction_comparison(reductions, large_words,
                               f'{model_name} — PCA vs t-SNE vs UMAP',
                               f'cat9_{model_name.lower()}_reduction_comparison.png')

    # Analogy evaluation
    print("\n[Analogy Evaluation...]")
    test_analogies = [
        ('man',   'king',   'woman',   'queen'),
        ('paris', 'france', 'berlin',  'germany'),
        ('paris', 'france', 'rome',    'italy'),
        ('man',   'actor',  'woman',   'actress'),
        ('good',  'better', 'bad',     'worse'),
    ]
    analogy_eval = evaluate_analogies(main_model, test_analogies)
    print(f"  Analogy accuracy: {analogy_eval['correct']}/{analogy_eval['total']} = {analogy_eval['accuracy']:.2%}")
    for r in analogy_eval['results']:
        status = '✓' if r['correct'] else '✗'
        print(f"    {status} {r['analogy']} → predicted: {r['predicted']}")

    plot_analogy_vectors(main_model, [(a, b, c, d) for a, b, c, d in test_analogies[:4]], model_name)

    # Similarity heatmap
    plot_similarity_heatmap(main_model, WORD_CATEGORIES, model_name)

    # Nearest neighbors
    print("\n[Nearest Neighbors...]")
    probe_words = ['king', 'science', 'government', 'war', 'city']
    nn_results  = {}
    for word in probe_words:
        if word in main_model.wv:
            nn = [(w, round(float(s), 4)) for w, s in main_model.wv.most_similar(word, topn=5)]
            nn_results[word] = nn
            print(f"  {word}: {[w for w, _ in nn]}")

    # Skip-gram comparison
    print("\n[Skip-gram vs CBOW Nearest Neighbors]")
    sg_model = models['skipgram_100']
    for word in ['king', 'government']:
        if word in sg_model.wv and word in main_model.wv:
            cbow_nn = [w for w, _ in main_model.wv.most_similar(word, topn=5)]
            sg_nn   = [w for w, _ in sg_model.wv.most_similar(word, topn=5)]
            print(f"  {word}:")
            print(f"    CBOW:      {cbow_nn}")
            print(f"    Skip-gram: {sg_nn}")

    all_metrics = {
        'models': {name: {'vocab_size': len(m.wv)} for name, m in models.items()},
        'analogy_eval': {k: v for k, v in analogy_eval.items() if k != 'results'},
        'nearest_neighbors': nn_results,
        'total_corpus_tokens': total_tokens,
        'category_words_found': len(category_words),
    }

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat9_word_embeddings.json'))
    print("\n  Metrics saved to outputs/metrics/cat9_word_embeddings.json")
    print("\n[CATEGORY 9 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
