"""
Category 8: Vector Semantics
==============================
Aim: Represent words as vectors and measure semantic similarity.
Objective: Implement TF-IDF vectorization, cosine similarity,
           and PMI (Pointwise Mutual Information).
Dataset: 20 Newsgroups + Brown Corpus
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
from nltk.corpus import brown

from utils.embeddings import compute_tfidf, compute_pmi, pairwise_cosine_matrix
from utils.preprocessing import clean_text, word_tokenize_nltk, remove_stopwords
from utils.evaluation import save_metrics

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)


def download_data():
    for r in ['brown', 'punkt', 'stopwords', 'punkt_tab']:
        nltk.download(r, quiet=True)


def load_newsgroups(n: int = 1000):
    """Load 20 Newsgroups if available, else build multi-category corpus from Reuters."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  categories=['sci.space', 'talk.politics.guns',
                                              'rec.sport.hockey', 'comp.graphics'])
        docs, labels = [], []
        for text, label in zip(data.data[:n], data.target[:n]):
            tokens = remove_stopwords(
                [t.lower() for t in word_tokenize_nltk(clean_text(text)) if t.isalpha() and len(t) > 2])
            if len(tokens) > 10:
                docs.append(tokens)
                labels.append(data.target_names[label])
        return docs, labels
    except Exception:
        # Fallback: use Reuters categories as proxy
        from nltk.corpus import reuters
        import nltk as _nltk
        _nltk.download('reuters', quiet=True)
        category_map = {
            'trade': 'trade', 'earn': 'earnings',
            'acq': 'acquisitions', 'money-fx': 'money',
        }
        docs, labels = [], []
        for cat, label in category_map.items():
            for fileid in reuters.fileids(cat)[:n // len(category_map)]:
                raw_tokens = [t.lower() for t in reuters.words(fileid)
                              if t.isalpha() and len(t) > 2]
                tokens = remove_stopwords(raw_tokens)
                if len(tokens) > 10:
                    docs.append(tokens)
                    labels.append(label)
        return docs[:n], labels[:n]


def load_brown_corpus(n: int = 2000):
    return [[t.lower() for t in s if t.isalpha() and len(t) > 2]
            for s in list(brown.sents())[:n]]


# ─── Manual TF-IDF ────────────────────────────────────────────────────────────

def compute_manual_tfidf(corpus: list) -> tuple:
    """Compute TF-IDF manually for demonstration."""
    N = len(corpus)
    df = collections.Counter()
    for doc in corpus:
        for word in set(doc):
            df[word] += 1

    tfidf_docs = []
    for doc in corpus:
        tf = collections.Counter(doc)
        tfidf = {}
        for word, count in tf.items():
            tf_val  = count / max(len(doc), 1)
            idf_val = math.log(N / max(df[word], 1))
            tfidf[word] = tf_val * idf_val
        tfidf_docs.append(tfidf)
    return tfidf_docs, df


def cosine_similarity_tfidf(tfidf1: dict, tfidf2: dict) -> float:
    """Cosine similarity between two TF-IDF vectors (as dicts)."""
    all_keys = set(tfidf1.keys()) | set(tfidf2.keys())
    dot   = sum(tfidf1.get(k, 0) * tfidf2.get(k, 0) for k in all_keys)
    norm1 = math.sqrt(sum(v*v for v in tfidf1.values()))
    norm2 = math.sqrt(sum(v*v for v in tfidf2.values()))
    return dot / max(norm1 * norm2, 1e-10)


# ─── Word-Context Matrix + PMI ────────────────────────────────────────────────

def build_word_context_matrix(corpus: list, vocab_size: int = 200, window: int = 2):
    """Build word-context co-occurrence matrix."""
    # Get top vocab
    counts = collections.Counter(tok for sent in corpus for tok in sent)
    vocab  = [w for w, _ in counts.most_common(vocab_size)]
    w2i    = {w: i for i, w in enumerate(vocab)}

    matrix = np.zeros((vocab_size, vocab_size))
    for sent in corpus:
        for i, word in enumerate(sent):
            if word not in w2i:
                continue
            start = max(0, i - window)
            end   = min(len(sent), i + window + 1)
            for j in range(start, end):
                if i != j and sent[j] in w2i:
                    matrix[w2i[word], w2i[sent[j]]] += 1

    # Compute PMI
    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    total    = matrix.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi_matrix = np.log2(
            np.where(matrix > 0,
                     (matrix * total) / (row_sums * col_sums + 1e-10),
                     1e-10))
        ppmi_matrix = np.maximum(pmi_matrix, 0)  # Positive PMI

    return ppmi_matrix, vocab, w2i


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_tfidf_heatmap(tfidf_matrix: np.ndarray, feature_names: list,
                       doc_labels: list, n_docs: int = 10, n_words: int = 20,
                       title: str = 'TF-IDF Heatmap'):
    matrix = tfidf_matrix[:n_docs, :n_words]
    labels_shown = doc_labels[:n_docs] if doc_labels else [str(i) for i in range(n_docs)]
    words_shown  = feature_names[:n_words]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(matrix, xticklabels=words_shown, yticklabels=labels_shown,
                cmap='YlOrRd', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Words')
    ax.set_ylabel('Documents')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    fname = 'cat8_tfidf_heatmap.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_doc_similarity_matrix(sim_matrix: np.ndarray, labels: list, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt='.2f',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_pmi_heatmap(ppmi_matrix: np.ndarray, vocab: list, top_n: int = 25):
    # Select top_n most frequent words
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(ppmi_matrix[:top_n, :top_n],
                xticklabels=vocab[:top_n], yticklabels=vocab[:top_n],
                cmap='viridis', ax=ax, fmt='.1f', annot=False)
    ax.set_title(f'PPMI Matrix — Top {top_n} Words (Brown Corpus)')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat8_ppmi_matrix.png'), dpi=150)
    plt.close()
    print("  Saved: cat8_ppmi_matrix.png")


def plot_word_cosine_similarity(sim_dict: dict, title: str, fname: str):
    """Bar chart of cosine similarities for a set of word pairs."""
    labels = list(sim_dict.keys())
    values = list(sim_dict.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in values]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_title(title)
    ax.set_xlabel('Cosine Similarity')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_top_tfidf_words(tfidf_matrix: np.ndarray, feature_names: list,
                          labels: list, n_per_class: int = 10):
    """Plot top TF-IDF words per category."""
    unique_labels = sorted(set(labels))
    n_cats = len(unique_labels)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5))
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, unique_labels):
        idx = [i for i, l in enumerate(labels) if l == cat]
        avg_vec = tfidf_matrix[idx].mean(axis=0)
        top_idx = avg_vec.argsort()[-n_per_class:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        top_vals  = [avg_vec[i] for i in top_idx]
        ax.barh(top_words[::-1], top_vals[::-1], color='steelblue')
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel('Avg TF-IDF')
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('Top TF-IDF Words per Category', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat8_tfidf_per_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cat8_tfidf_per_category.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 8: VECTOR SEMANTICS")
    print("=" * 60)

    download_data()
    all_metrics = {}

    # ── TF-IDF on 20 Newsgroups ──────────────────────────────────────────────
    print("\n[TF-IDF on 20 Newsgroups]")
    docs, labels = load_newsgroups(n=800)
    print(f"  Loaded {len(docs)} documents, {len(set(labels))} categories")

    tfidf_matrix, feature_names = compute_tfidf(docs)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # Top words globally
    avg_tfidf = tfidf_matrix.mean(axis=0)
    top_idx   = avg_tfidf.argsort()[-20:][::-1]
    top_words = [(feature_names[i], round(float(avg_tfidf[i]), 4)) for i in top_idx]
    print(f"  Top 10 words globally: {top_words[:10]}")

    # Document similarity via TF-IDF + cosine
    n_sample = min(20, len(docs))
    sim_matrix = pairwise_cosine_matrix(tfidf_matrix[:n_sample])
    sample_labels = [f"{labels[i][:10]}_{i}" for i in range(n_sample)]

    plot_tfidf_heatmap(tfidf_matrix, feature_names, labels, n_docs=12, n_words=25)
    plot_doc_similarity_matrix(sim_matrix[:12, :12], sample_labels[:12],
                                'Document Cosine Similarity (TF-IDF)',
                                'cat8_doc_similarity.png')
    plot_top_tfidf_words(tfidf_matrix, feature_names, labels)

    all_metrics['tfidf'] = {
        'n_docs': len(docs),
        'vocab_size': len(feature_names),
        'top_words': top_words[:10],
    }

    # ── Cosine Similarity Analysis ───────────────────────────────────────────
    print("\n[Cosine Similarity Analysis]")
    manual_tfidf, df = compute_manual_tfidf(docs[:200])

    # Compute word-level TF-IDF vectors across documents
    # and find most/least similar document pairs
    intra_class_sims = {}
    inter_class_sims = {}
    for i in range(min(50, len(docs))):
        for j in range(i + 1, min(50, len(docs))):
            sim = cosine_similarity_tfidf(manual_tfidf[i], manual_tfidf[j])
            key = (i, j)
            if labels[i] == labels[j]:
                intra_class_sims[key] = sim
            else:
                inter_class_sims[key] = sim

    avg_intra = sum(intra_class_sims.values()) / max(len(intra_class_sims), 1)
    avg_inter = sum(inter_class_sims.values()) / max(len(inter_class_sims), 1)
    print(f"  Average intra-class cosine similarity: {avg_intra:.4f}")
    print(f"  Average inter-class cosine similarity: {avg_inter:.4f}")

    # Compare specific document pairs
    sim_comparison = {
        'Avg Intra-class': round(avg_intra, 4),
        'Avg Inter-class': round(avg_inter, 4),
    }
    plot_word_cosine_similarity(sim_comparison,
                                'Intra vs Inter-class Document Similarity',
                                'cat8_intra_inter_similarity.png')

    all_metrics['cosine'] = {
        'avg_intra_class': round(avg_intra, 4),
        'avg_inter_class': round(avg_inter, 4),
    }

    # ── PMI Analysis ─────────────────────────────────────────────────────────
    print("\n[PMI Analysis on Brown Corpus]")
    brown_corpus = load_brown_corpus(n=3000)
    ppmi_matrix, vocab, w2i = build_word_context_matrix(brown_corpus,
                                                         vocab_size=200, window=3)
    print(f"  PPMI matrix: {ppmi_matrix.shape}")
    print(f"  Vocabulary: {len(vocab)} words")

    # Top PMI word pairs
    flat_idx = np.argsort(ppmi_matrix.flatten())[::-1][:50]
    top_pairs = []
    for idx in flat_idx:
        i, j = divmod(idx, len(vocab))
        if i != j and ppmi_matrix[i, j] > 0:
            top_pairs.append((vocab[i], vocab[j], round(float(ppmi_matrix[i, j]), 3)))
    top_pairs = top_pairs[:10]
    print(f"\n  Top 10 PPMI word pairs:")
    for w1, w2, pmi_val in top_pairs:
        print(f"    {w1:<12} ↔ {w2:<12}  PPMI={pmi_val:.3f}")

    # Semantic similarity via PPMI vectors
    semantic_pairs = [('man', 'woman'), ('city', 'town'), ('good', 'bad'),
                       ('king', 'president'), ('car', 'truck')]
    ppmi_similarities = {}
    for w1, w2 in semantic_pairs:
        if w1 in w2i and w2 in w2i:
            v1 = ppmi_matrix[w2i[w1]]
            v2 = ppmi_matrix[w2i[w2]]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(v1, v2) / (norm1 * norm2))
                ppmi_similarities[f"{w1}-{w2}"] = round(sim, 4)

    print(f"\n  Semantic similarities (PPMI vectors):")
    for pair, sim in ppmi_similarities.items():
        print(f"    {pair}: {sim:.4f}")

    plot_pmi_heatmap(ppmi_matrix, vocab, top_n=20)
    plot_word_cosine_similarity(ppmi_similarities,
                                'Semantic Similarity (PPMI Vectors)',
                                'cat8_ppmi_word_similarity.png')

    all_metrics['pmi'] = {
        'matrix_shape': list(ppmi_matrix.shape),
        'vocab_size': len(vocab),
        'top_pairs': top_pairs,
        'semantic_similarities': ppmi_similarities,
    }

    save_metrics(all_metrics, os.path.join(METRICS_DIR, 'cat8_vector_semantics.json'))
    print("\n  Metrics saved to outputs/metrics/cat8_vector_semantics.json")
    print("\n[CATEGORY 8 COMPLETE]")
    return all_metrics


if __name__ == '__main__':
    run()
