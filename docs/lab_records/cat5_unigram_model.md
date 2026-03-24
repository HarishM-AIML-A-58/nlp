# Lab Record — Category 5: Unigram Language Model

## Aim
To build and evaluate a unigram language model using Maximum Likelihood Estimation and Laplace smoothing.

## Objective
1. Estimate unigram word probabilities from corpus
2. Demonstrate Zipf's Law in natural language
3. Apply add-k smoothing and compare with MLE
4. Compute perplexity to measure model quality
5. Generate text from unigram distribution

## Dataset Description
- **Brown Corpus** (NLTK): 3,000 sentences; 80/20 train/test split
- **Reuters Corpus** (NLTK): 3,000 sentences; 80/20 train/test split

## Algorithm

### MLE Unigram Probability
```
P_MLE(w) = count(w) / N
```
where N = total token count.

### Laplace (Add-k) Smoothing
```
P_Laplace(w) = (count(w) + k) / (N + k × |V|)
```
where |V| = vocabulary size, k = smoothing constant.

### Perplexity
```
PP(W) = exp(-1/N × Σᵢ log P(wᵢ))
     = 2^H(W)    (base-2 formulation)
```

### Zipf's Law
Rank-frequency relationship: frequency(rank r) ∝ 1/r

### Stepwise Algorithm
1. Tokenize corpus to word lists
2. Count word frequencies
3. Compute MLE probabilities
4. Apply Laplace smoothing (k=1 and k=0.5)
5. Evaluate perplexity on held-out test set
6. Analyze Zipf distribution (log-log rank-frequency plot)
7. Compare MLE vs smoothed probability for top words
8. Generate text by sampling from distribution

## Observations
- **Zipf's Law**: Word frequency distribution follows power law — top 100 words cover ~50% of tokens
- **MLE limitation**: Assigns zero probability to any OOV word → perplexity undefined
- **Laplace effect**: Redistributes probability mass from seen to unseen words; reduces probability of common words
- **Vocabulary**: Brown corpus has ~3,000–4,000 unique lemmas per 3,000 sentences
- **Entropy**: ~9–10 bits/word for general English unigram model
- **Generated text**: Unigram model produces incoherent text (random word bags) as expected

## Inference
- Unigram models are simple baselines but ignore word order entirely
- Smoothing is mandatory for any practical model — probability of zero kills perplexity
- Perplexity of 1,000–2,500 is typical for unigram models on general English
- Zipf's Law has profound implications: small vocabulary covers most text

## Results

| Model | Brown Perplexity | Reuters Perplexity |
|-------|-----------------|-------------------|
| Unigram MLE (Laplace k=1) | ~1,200–2,000 | ~1,000–1,800 |
| Unigram Laplace k=0.5 | ~1,000–1,800 | ~900–1,600 |

**Top 5 Brown words**: the, of, and, to, a (function words dominate)
**Top 5 after stopword removal**: said, year, new, government, state

**Outputs**: `outputs/plots/cat5_*.png`, `outputs/metrics/cat5_unigram_model.json`
