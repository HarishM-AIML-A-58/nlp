# Lab Record — Category 10: Hidden Markov Model for POS Tagging

## Aim

To implement a Hidden Markov Model (HMM) entirely from scratch using NumPy for Part-of-Speech (POS) tagging, and to demonstrate the core HMM inference algorithms — Forward, Backward, Viterbi, and Baum-Welch — on the Brown Corpus.

---

## Objective

1. Load and preprocess the Brown Corpus with the Universal POS tagset.
2. Estimate HMM parameters (π, A, B) from labelled training data using Maximum Likelihood Estimation with Laplace smoothing.
3. Implement the **Forward Algorithm** to compute the log-probability of an observation sequence given the model.
4. Implement the **Backward Algorithm** and compute per-state posteriors using the Forward-Backward product.
5. Implement the **Viterbi Algorithm** to decode the most probable tag sequence for unseen sentences.
6. Implement a simplified **Baum-Welch (EM)** procedure to re-estimate parameters from unlabelled data.
7. Evaluate Viterbi tagging accuracy overall and per POS tag.
8. Visualise tag distributions, transition structure, emission probabilities, and algorithm outputs.

---

## Dataset Description

| Property            | Value                              |
|---------------------|------------------------------------|
| Corpus              | Brown Corpus (NLTK)                |
| Tagset              | Universal POS (12 coarse tags)     |
| Sentences used      | First 3 000 (for speed)            |
| Train / Test split  | 80% / 20%  (~2 383 / ~596 sents)   |
| Vocabulary size     | ~8 388 unique word types           |
| Tags (N)            | 12                                 |

**Universal tagset labels:** `.` (punctuation), `ADJ`, `ADP`, `ADV`, `CONJ`, `DET`, `NOUN`, `NUM`, `PRON`, `PRT`, `VERB`, `X`.

---

## Algorithm

### HMM Components

A first-order HMM is parameterised by three matrices estimated from labelled sentences:

| Symbol | Dimension | Meaning                         |
|--------|-----------|---------------------------------|
| **π**  | (N,)      | Initial state probability       |
| **A**  | (N × N)   | Transition probability A[i,j] = P(tag_j \| tag_i) |
| **B**  | (N × V)   | Emission probability B[i,w] = P(word_w \| tag_i)  |

All counts are smoothed with Laplace smoothing (k = 1) before normalisation to avoid zero probabilities.

### Forward Algorithm

The Forward Algorithm computes **α_t(i) = P(o₁…o_t, q_t = i | λ)** via dynamic programming in log-space to prevent numerical underflow.

1. **Initialise**: `α₀(i) = log π(i) + log B(i, o₀)`
2. **Recurse**: `α_t(j) = log B(j, o_t) + log-sum-exp_i [ α_{t-1}(i) + log A(i,j) ]`
3. **Terminate**: `log P(O|λ) = log-sum-exp_i [ α_{T-1}(i) ]`

Returns the total log-probability of the observation sequence.

### Backward Algorithm

The Backward Algorithm computes **β_t(i) = P(o_{t+1}…o_T | q_t = i, λ)** in log-space:

1. **Initialise**: `β_{T-1}(i) = log(1) = 0`
2. **Recurse** (right to left): `β_t(i) = log-sum-exp_j [ log A(i,j) + log B(j, o_{t+1}) + β_{t+1}(j) ]`

Combined with the Forward probabilities, the **state posteriors** are:

```
γ_t(i) = α_t(i) + β_t(i)  (log-space, then normalised per time step)
```

### Viterbi Algorithm

Viterbi finds the single best tag path **q* = argmax_Q P(Q, O | λ)**:

1. **Initialise**: `δ₀(i) = log π(i) + log B(i, o₀)`
2. **Recurse**: `δ_t(j) = max_i [ δ_{t-1}(i) + log A(i,j) ] + log B(j, o_t)`
3. **Backtrack**: follow the argmax pointers from the best final state.

### Baum-Welch (EM)

A simplified two-iteration Baum-Welch procedure re-estimates π, A, and B from **unlabelled** test sentences:

1. **E-step**: run Forward + Backward on each observation sequence; compute γ (state posteriors) and ξ (pairwise posteriors).
2. **M-step**: re-normalise accumulated sufficient statistics to obtain new π, A, B.

After each iteration the average log P(sentence|model) is reported to confirm the likelihood increases.

---

## Observations

- The Universal tagset collapses Brown's fine-grained tags into 12 coarse categories, making parameter estimation robust on only 3 000 sentences.
- **Frequent, unambiguous tags** (`.`, `CONJ`, `DET`, `ADP`) achieve near-perfect Viterbi accuracy because their emission distributions are peaked and transitions are consistent.
- **Ambiguous open-class tags** (`ADJ`, `ADV`, `NUM`, `X`) have lower accuracy due to high lexical overlap with `NOUN` and `VERB`.
- The Forward Algorithm log-probabilities scale roughly linearly with sentence length, confirming correct log-space implementation.
- The Backward-derived state posteriors agree with Viterbi on unambiguous tokens and diverge slightly on ambiguous ones, where soft assignment spreads probability across tags.
- Two Baum-Welch EM iterations raise the average log-likelihood of the test corpus from ≈ −119.6 to ≈ −92.2, demonstrating parameter adaptation to the unlabelled domain.

---

## Inference

1. An HMM trained with simple frequency counts and Laplace smoothing on fewer than 2 500 sentences already achieves **≈ 87% POS tagging accuracy** on the Universal tagset — a strong baseline.
2. Working in **log-space** throughout (log-sum-exp trick) is essential; naive probability products underflow to zero even for 10-word sentences.
3. **OOV words** are handled by treating their emission probability as uniform (log B = 0 for all states), which is a soft fallback; more sophisticated handling (morphological features, suffix heuristics) would further improve accuracy.
4. Baum-Welch shows that unsupervised EM can adapt model parameters to new data without labels, though convergence with only 2 iterations is partial; more iterations and careful initialisation would be needed for a fully unsupervised system.
5. The transition heatmap reveals linguistically sensible patterns: determiners almost always precede nouns, punctuation follows verbs or nouns, and prepositions are followed by noun phrases — all captured automatically from data.

---

## Results

### Viterbi POS Tagging Accuracy

| POS Tag | Accuracy |
|---------|----------|
| `.`     | 1.0000   |
| `CONJ`  | 1.0000   |
| `DET`   | 0.9867   |
| `ADP`   | 0.9865   |
| `PRON`  | 0.9091   |
| `NOUN`  | 0.8725   |
| `VERB`  | 0.7715   |
| `PRT`   | 0.6769   |
| `NUM`   | 0.6383   |
| `ADJ`   | 0.6192   |
| `ADV`   | 0.5755   |
| `X`     | 0.0000   |
| **Overall** | **0.8723** |

### Forward Algorithm — Sample Log-Probabilities

| Sentence | Length | log P(sent \| model) |
|----------|--------|----------------------|
| S1       | 31     | −204.74              |
| S2       | 25     | −163.27              |
| S5       | 15     | −112.19              |
| S8       | 3      | −16.83               |
| S13      | 6      | −27.07               |

### Baum-Welch EM Convergence

| Iteration | Avg. log P(sentence \| model) |
|-----------|-------------------------------|
| Before    | —                             |
| After 1   | −119.58                       |
| After 2   | −92.19                        |

### HMM Model Summary

| Parameter       | Value   |
|-----------------|---------|
| Number of tags  | 12      |
| Vocabulary size | 8 388   |
| Training sents  | ~2 383  |
| Test sents      | ~596    |
| Smoothing       | Laplace (k=1) |
| BW iterations   | 2       |
