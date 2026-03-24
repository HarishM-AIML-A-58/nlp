# Lab Record — Category 8: Vector Semantics

## Aim
To represent documents and words as numerical vectors and measure semantic similarity using statistical methods.

## Objective
1. Build TF-IDF document-term matrix
2. Compute cosine similarity between documents
3. Compute word co-occurrence matrix and PPMI scores
4. Analyze intra-class vs inter-class similarity

## Dataset Description
- **20 Newsgroups** (sklearn): 800 documents from 4 categories (sci.space, politics, hockey, graphics)
- **Brown Corpus** (NLTK): 3,000 sentences for PPMI word matrix

## Algorithm

### TF-IDF
```
TF(t, d) = count(t, d) / |d|
IDF(t) = log(N / df(t))
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### Cosine Similarity
```
cos(A, B) = (A · B) / (|A| × |B|)
          = Σ aᵢbᵢ / (√Σaᵢ² × √Σbᵢ²)
```

### Pointwise Mutual Information (PMI)
```
PMI(w₁, w₂) = log₂(P(w₁, w₂) / (P(w₁) × P(w₂)))
```

### Positive PMI (PPMI)
```
PPMI(w₁, w₂) = max(PMI(w₁, w₂), 0)
```

### Stepwise Algorithm
1. Load 20 Newsgroups; tokenize, remove stopwords
2. Compute TF-IDF matrix (sklearn) — 5,000 features
3. Compute pairwise cosine similarity between documents
4. Compute average intra-class vs inter-class cosine similarity
5. Build word-context co-occurrence matrix (Brown, window=3)
6. Compute PPMI matrix for top-200 vocabulary words
7. Visualize: TF-IDF heatmap, PPMI matrix, similarity comparisons

## Observations
- **TF-IDF**: Effectively captures discriminative terms per category — "shuttle" for sci.space, "gun" for politics.guns
- **Intra-class similarity**: Documents from same category are consistently more similar than cross-category
- **Inter-class similarity**: Very low (~0.05–0.15) confirming TF-IDF captures topic differences
- **PPMI matrix**: High scores for strongly associated word pairs (e.g., "new" ↔ "york", "united" ↔ "states")
- **Word vectors via PPMI**: More interpretable than neural embeddings; explicit features show why words are similar
- **Sparsity**: TF-IDF matrix is ~99% sparse; PPMI matrix is ~80% zero

## Inference
- TF-IDF + cosine similarity is a strong baseline for document similarity without any training
- PPMI vectors capture associative word meaning from raw co-occurrence statistics
- The gap between intra and inter-class similarity validates that vector representations cluster by topic
- PPMI is less affected by word frequency bias compared to raw co-occurrence counts

## Results

| Metric | Value |
|--------|-------|
| TF-IDF vocab size | ~5,000 features |
| Avg intra-class cosine similarity | ~0.10–0.25 |
| Avg inter-class cosine similarity | ~0.02–0.08 |
| PPMI matrix size | 200 × 200 |

**Top PPMI word pairs** (Brown Corpus):
- "new" ↔ "york", "united" ↔ "states", "per" ↔ "cent"

**Outputs**: `outputs/plots/cat8_*.png`, `outputs/metrics/cat8_vector_semantics.json`
