# Lab Record — Category 9: Word Embeddings + Visualization

## Aim
To train dense word embedding models and visualize the learned semantic structure using dimensionality reduction.

## Objective
1. Train Word2Vec (CBOW and Skip-gram) on combined corpus
2. Reduce 100-dimensional embeddings to 2D using PCA, t-SNE, and UMAP
3. Visualize semantic word clusters
4. Evaluate using word analogies
5. Compare PCA vs t-SNE vs UMAP for embedding visualization

## Dataset Description
- **Brown Corpus** (NLTK): 8,000 sentences (~160k tokens)
- **Reuters Corpus** (NLTK): 5,000 sentences (~100k tokens)
- **Combined**: ~260k tokens total

## Algorithm

### Word2Vec Architecture

**CBOW**:
```
Input: context window words → hidden: average of word vectors → output: center word
Loss: cross-entropy over vocabulary (with negative sampling)
```

**Skip-gram**:
```
Input: center word → output: each context word
Loss: Σ log P(w_context | w_center)
```

### PCA (Principal Component Analysis)
```
Find k eigenvectors of covariance matrix corresponding to k largest eigenvalues.
Project: X_reduced = X × W_k
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```
1. Compute pairwise similarities in high-D space (Gaussian)
2. Compute pairwise similarities in low-D space (Student-t)
3. Minimize KL divergence between distributions via gradient descent
```

### UMAP (Uniform Manifold Approximation and Projection)
```
1. Construct fuzzy topological representation of high-D data
2. Optimize low-D representation to match topological structure
3. Preserves both local and global structure better than t-SNE
```

### Hyperparameters
- Vector size: 100
- Window: 5, Min count: 3, Epochs: 15
- t-SNE: perplexity=15, iterations=1,000
- UMAP: n_neighbors=10, min_dist=0.1

## Observations
- **Semantic clustering**: Countries, cities, colors, and royalty words form distinct clusters in all reduction methods
- **PCA**: Fast, linear — preserves global structure; clusters overlap more
- **t-SNE**: Non-linear — tight, well-separated clusters; best for discovering local neighborhoods
- **UMAP**: Preserves both local and global structure; fastest of the non-linear methods
- **Analogy accuracy**: king - man + woman ≈ queen works when all four words have sufficient training examples
- **Similarity patterns**: Antonyms (good/bad) appear close in embedding space (they appear in similar contexts)
- **CBOW vs Skip-gram**: CBOW produces more compact clusters; Skip-gram shows more spread (captures rarer words better)

## Inference
- Word embeddings encode syntactic and semantic relationships as geometric operations in vector space
- t-SNE is best for visualizing embedding clusters but doesn't preserve global structure
- UMAP is recommended for production visualizations: faster and preserves more structure
- Analogy tasks require large corpora for reliable results; ~260k tokens is borderline
- Categorical semantic groups (countries, animals) are reliably separated across all reduction methods

## Results

**Analogy Test** (CBOW-100):
| Analogy | Predicted | Correct? |
|---------|-----------|----------|
| king - man + woman | queen/princess | Varies |
| paris - france + germany | berlin | Varies |

**Word Similarity** (CBOW-100):
| Category | Avg Intra-group Similarity |
|----------|--------------------------|
| Countries | ~0.45–0.65 |
| Animals | ~0.40–0.60 |
| Colors | ~0.35–0.55 |
| Royalty | ~0.50–0.70 |

**Model Vocab Sizes**:
- CBOW-100: ~4,000–6,000 words
- Skip-gram-100: ~4,000–6,000 words
- CBOW-50: ~4,000–6,000 words

**Outputs**: `outputs/plots/cat9_*.png`, `outputs/metrics/cat9_word_embeddings.json`
