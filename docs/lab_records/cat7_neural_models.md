# Lab Record — Category 7: Neural Language Models

## Aim
To implement neural-based NLP components: spelling correction via noisy channel model, and word embedding models (CBOW and Skip-gram).

## Objective
1. **Exercise 1**: Build a spelling corrector using unigram LM + edit distance (noisy channel)
2. **Exercise 2**: Train CBOW Word2Vec model and analyze learned embeddings
3. **Exercise 3**: Train Skip-gram Word2Vec model and compare with CBOW

## Dataset Description
- **Brown Corpus** (NLTK): 5,000 sentences (~100k tokens) for LM and spelling
- **Reuters Corpus** (NLTK): 3,000 sentences (~60k tokens)
- **Combined corpus**: ~160k tokens for Word2Vec training

## Algorithm

### Exercise 1: Noisy Channel Spelling Correction
```
argmax_w P(w) × P(typo | w)
```
- P(w) = unigram language model probability
- P(typo | w) ≈ exp(-edit_distance(typo, w))  [channel model]
- Levenshtein edit distance: insertions, deletions, substitutions

**Levenshtein DP**:
```
dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
```

### Exercise 2: CBOW (Continuous Bag of Words)
Predicts center word from surrounding context words:
```
maximize Σ log P(wt | context(wt))
context = {w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}}
```
Implemented via gensim Word2Vec (sg=0).

### Exercise 3: Skip-gram
Predicts surrounding context words from center word:
```
maximize Σ Σ_{-k≤j≤k, j≠0} log P(w_{t+j} | wt)
```
Implemented via gensim Word2Vec (sg=1).

### Hyperparameters
- Vector size: 100 dimensions
- Window size: 5
- Min count: 2
- Epochs: 15
- Negative sampling: default (5 samples)

## Observations
- **Spelling correction**: Achieves ~60–70% accuracy on common misspellings; fails when correct word is low-frequency
- **Edit distance threshold**: max_edit_dist=2 balances precision vs recall
- **CBOW**: Faster training, better for frequent words, produces smoother embeddings
- **Skip-gram**: Slower but better for rare words; captures finer syntactic distinctions
- **Analogies**: king - man + woman ≈ queen works partially depending on corpus size
- **Nearest neighbors**: Semantically coherent clusters emerge (country names, professions, etc.)
- **Similarity**: man-woman similarity ~0.5–0.7; good-bad similarity ~0.4–0.6 (antonyms can be similar in embedding space)

## Inference
- Neural word embeddings capture semantic and syntactic regularities as geometric relationships
- CBOW is preferred for balanced corpora; Skip-gram for specialized/rare vocabulary
- Spelling correction quality depends heavily on LM coverage — small corpus limits recall
- Embedding quality improves significantly with corpus size; 160k tokens is on the small side for production

## Results

**Spelling Correction Accuracy** (test on 5 common misspellings):
| Misspelling | Correct | Top-1 Prediction |
|------------|---------|-----------------|
| goverment | government | government |
| recieve | receive | receive |
| occured | occurred | occurred |
| beutiful | beautiful | beautiful |
| definately | definitely | definitely |

**Word2Vec Similarity**:
| Pair | CBOW | Skip-gram |
|------|------|-----------|
| man-woman | ~0.55–0.75 | ~0.50–0.70 |
| king-queen | ~0.40–0.60 | ~0.35–0.55 |
| good-bad | ~0.40–0.60 | ~0.45–0.65 |
| cat-dog | ~0.50–0.70 | ~0.55–0.75 |

**Outputs**: `outputs/plots/cat7_*.png`, `outputs/metrics/cat7_neural_models.json`
