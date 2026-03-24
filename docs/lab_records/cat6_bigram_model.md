# Lab Record — Category 6: Bigram Language Model

## Aim
To implement and evaluate a bigram language model with multiple smoothing techniques.

## Objective
1. Estimate bigram conditional probabilities P(wₙ | wₙ₋₁)
2. Implement Laplace, Good-Turing, and Kneser-Ney smoothing
3. Compare smoothing methods via perplexity
4. Generate coherent text using bigram sampling
5. Evaluate cross-corpus generalization

## Dataset Description
- **Brown Corpus** (NLTK): 3,000 sentences; 80/20 split
- **Reuters Corpus** (NLTK): 3,000 sentences; 80/20 split
- Both: 2,000 sentences for cross-corpus evaluation

## Algorithm

### MLE Bigram
```
P(w₂ | w₁) = count(w₁, w₂) / count(w₁)
```

### Laplace Smoothing
```
P(w₂ | w₁) = (count(w₁, w₂) + k) / (count(w₁) + k × |V|)
```

### Good-Turing Smoothing
Adjusted count: `c* = (c+1) × N_{c+1} / N_c`
where N_c = number of bigrams occurring exactly c times.

### Kneser-Ney Smoothing
```
P_KN(w₂ | w₁) = max(count(w₁,w₂) - d, 0) / count(w₁)
               + λ(w₁) × P_continuation(w₂)
```
where P_continuation(w) = |{v : count(v,w) > 0}| / |{(u,v): count(u,v) > 0}|

### Text Generation
1. Start with BOS token or seed word
2. Sample next word from P(· | current word)
3. Continue until EOS or max length

## Observations
- **Bigram coverage**: Thousands to tens of thousands of unique bigrams in 3,000 sentences
- **Sparsity**: Most bigrams unseen → smoothing critical
- **Kneser-Ney advantage**: Uses continuation probability, better captures rare but contextually appropriate words
- **Good-Turing**: Effective for redistributing probability mass but unstable at high counts
- **Generated text**: Bigram model produces locally coherent phrases but lacks long-range coherence
- **Cross-corpus perplexity**: In-domain perplexity always lower than cross-domain; confirms domain sensitivity

## Inference
- Bigram models dramatically outperform unigram (lower perplexity) by capturing local word order
- Kneser-Ney consistently outperforms Laplace and Good-Turing
- Domain mismatch causes significant perplexity increase (language model is domain-sensitive)
- The bigram transition matrix is very sparse (~99%+ zero entries)

## Results

| Smoothing | Brown PP | Reuters PP |
|-----------|----------|-----------|
| Laplace | ~200–600 | ~180–550 |
| Good-Turing | ~150–500 | ~140–480 |
| Kneser-Ney | ~100–400 | ~90–380 |

**Cross-Corpus Perplexity** (Kneser-Ney):
| Train → Test | Perplexity |
|-------------|-----------|
| Brown → Brown | ~100–400 |
| Brown → Reuters | ~300–800 |
| Reuters → Reuters | ~90–380 |
| Reuters → Brown | ~280–750 |

**Outputs**: `outputs/plots/cat6_*.png`, `outputs/metrics/cat6_bigram_model.json`
