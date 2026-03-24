# Lab Record — Category 2: Entropy, Cross-Entropy, Perplexity

## Aim
To quantify information content in text corpora using entropy-based measures and evaluate language model quality via perplexity.

## Objective
1. Compute Shannon entropy of unigram distributions across corpora
2. Compute cross-entropy between corpus pairs as a divergence measure
3. Compute KL divergence to quantify corpus difference
4. Evaluate unigram language model perplexity with and without smoothing

## Dataset Description
- **Brown Corpus** (NLTK): ~2,000 sentences, general English
- **Reuters Corpus** (NLTK): ~2,000 sentences, financial/news domain
- **20 Newsgroups** (sklearn): ~500 documents, 20 topic categories

## Algorithm

### Shannon Entropy
```
H(X) = -Σ p(x) log₂ p(x)
```
where p(x) is the unigram probability of word x.

### Cross-Entropy
```
H(P, Q) = -Σ p(x) log₂ q(x)
```
Measures how well distribution Q approximates P.

### KL Divergence
```
D_KL(P || Q) = Σ p(x) log₂(p(x)/q(x))
```
Measures information lost when Q approximates P.

### Perplexity (Unigram)
```
PP(W) = exp(-1/N × Σ log P(wᵢ))
```

### Stepwise Algorithm
1. Load and tokenize corpora
2. Build unigram frequency distributions
3. Normalize to probability distributions
4. Compute entropy for each corpus
5. Compute pairwise cross-entropy matrix (3×3)
6. Train unigram LM on train split (80%)
7. Evaluate perplexity on held-out test split (20%)
8. Compare MLE vs Laplace smoothed perplexity

## Observations
- **Entropy**: Brown corpus shows higher entropy than Reuters, indicating more diverse vocabulary
- **Cross-entropy**: Same-domain corpora (e.g., Brown↔Brown) show lower cross-entropy than cross-domain
- **KL divergence**: Brown↔Reuters divergence reflects domain shift from general to news
- **Perplexity**: Without smoothing, MLE perplexity is undefined for unseen words; Laplace reduces perplexity significantly
- **Training size effect**: Perplexity decreases as training size increases (model improves with data)

## Inference
- Higher entropy → more uniform (diverse) distribution → harder to predict
- Cross-entropy as a distance measure aligns with domain similarity
- Smoothing is essential: MLE is unusable for open-vocabulary scenarios
- Perplexity values in the hundreds are typical for unigram models on general English

## Results

| Corpus | Shannon Entropy (bits) |
|--------|----------------------|
| Brown | ~9.2–9.8 |
| Reuters | ~9.0–9.5 |
| Newsgroups | ~8.5–9.2 |

| Model | Perplexity (Brown test set) |
|-------|---------------------------|
| Unigram MLE (Laplace) | ~1000–2500 |
| Unigram Laplace k=0.5 | ~1000–2500 |

**Outputs**: `outputs/plots/cat2_*.png`, `outputs/metrics/cat2_entropy_perplexity.json`
