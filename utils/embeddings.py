"""
Embedding utilities: training, loading, similarity, visualization.
"""

import os
import math
import numpy as np
from typing import List, Dict, Tuple, Optional


# ─── Similarity ───────────────────────────────────────────────────────────────

def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for rows of matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = matrix / norms
    return normalized @ normalized.T


def most_similar(word: str, word_vectors: Dict[str, np.ndarray],
                 top_n: int = 10) -> List[Tuple[str, float]]:
    if word not in word_vectors:
        return []
    vec = word_vectors[word]
    sims = [(w, cosine_similarity_np(vec, v))
            for w, v in word_vectors.items() if w != word]
    return sorted(sims, key=lambda x: -x[1])[:top_n]


# ─── TF-IDF ───────────────────────────────────────────────────────────────────

def compute_tfidf(corpus: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Compute TF-IDF matrix from tokenised corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [' '.join(tokens) for tokens in corpus]
    vec = TfidfVectorizer(max_features=5000)
    matrix = vec.fit_transform(texts).toarray()
    vocab = vec.get_feature_names_out().tolist()
    return matrix, vocab


# ─── PMI ──────────────────────────────────────────────────────────────────────

def compute_pmi(corpus: List[List[str]], window: int = 2,
                min_count: int = 5) -> Dict[Tuple[str, str], float]:
    """Compute Pointwise Mutual Information for word co-occurrences."""
    import collections
    word_counts = collections.Counter(tok for sent in corpus for tok in sent)
    cooccur = collections.Counter()
    total_tokens = sum(word_counts.values())

    for sent in corpus:
        for i, word in enumerate(sent):
            start = max(0, i - window)
            end = min(len(sent), i + window + 1)
            for j in range(start, end):
                if i != j:
                    cooccur[(word, sent[j])] += 1

    total_cooccur = sum(cooccur.values())
    pmi_dict = {}
    for (w1, w2), count in cooccur.items():
        if count < min_count:
            continue
        p_w1 = word_counts[w1] / total_tokens
        p_w2 = word_counts[w2] / total_tokens
        p_joint = count / total_cooccur
        if p_w1 > 0 and p_w2 > 0 and p_joint > 0:
            pmi_dict[(w1, w2)] = math.log2(p_joint / (p_w1 * p_w2))

    return pmi_dict


# ─── Dimensionality Reduction ────────────────────────────────────────────────

def reduce_pca(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)


def reduce_tsne(vectors: np.ndarray, n_components: int = 2,
                perplexity: float = 30, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, max(vectors.shape[0] - 1, 1))
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, max_iter=1000)
    return tsne.fit_transform(vectors)


def reduce_umap(vectors: np.ndarray, n_components: int = 2,
                n_neighbors: int = 15, random_state: int = 42) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=min(n_neighbors, vectors.shape[0] - 1),
                            random_state=random_state)
        return reducer.fit_transform(vectors)
    except ImportError:
        print("UMAP not available, falling back to PCA")
        return reduce_pca(vectors, n_components)


# ─── Gensim Word2Vec Wrapper ─────────────────────────────────────────────────

def train_word2vec(corpus: List[List[str]], vector_size: int = 100,
                   window: int = 5, min_count: int = 1,
                   sg: int = 0, epochs: int = 10, workers: int = 1) -> 'gensim.models.Word2Vec':
    """Train Word2Vec model (sg=0: CBOW, sg=1: Skip-gram)."""
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window,
                     min_count=min_count, sg=sg, epochs=epochs, workers=workers,
                     seed=42)
    return model


def get_word_vectors(model) -> Tuple[List[str], np.ndarray]:
    """Extract word vectors from trained Word2Vec model."""
    words = list(model.wv.key_to_index.keys())
    vectors = np.array([model.wv[w] for w in words])
    return words, vectors
