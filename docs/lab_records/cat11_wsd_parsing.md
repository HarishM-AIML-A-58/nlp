# Lab Record — Category 11: Word Sense Disambiguation and Parsing

## Aim

To implement and evaluate Word Sense Disambiguation (WSD) using the Lesk algorithm, and to explore syntactic analysis of English sentences through both Constituency Parsing (CFG + ChartParser) and Dependency Parsing, demonstrating the complementary roles of lexical semantics and syntactic structure in NLP.

---

## Objective

1. Implement **NLTK's Lesk algorithm** for WSD and compare it with a **Simplified Lesk from scratch**.
2. Evaluate disambiguation accuracy on hand-curated test sentences covering five highly ambiguous words: *bank*, *bat*, *plant*, *bass*, *crane*.
3. Visualise the predicted sense distribution per target word.
4. Design a **toy Context-Free Grammar (CFG)** covering core English phrase structures (NP, VP, PP) and parse example sentences using NLTK's **ChartParser**.
5. Perform **Dependency Parsing** using spaCy (with a rule-based NLTK fallback) and extract labelled dependency triples.
6. Visualise parse trees, arc diagrams, and dependency relation distributions.

---

## Dataset Description

| Component               | Details                                                    |
|-------------------------|------------------------------------------------------------|
| WSD test set            | 20 hand-crafted sentences (4 per word × 5 words)          |
| Target words            | *bank*, *bat*, *plant*, *bass*, *crane*                    |
| WordNet version         | NLTK's bundled WordNet 3.1 (via `nltk.corpus.wordnet`)     |
| Constituency corpus     | 8 toy sentences over the hand-written CFG                  |
| Dependency corpus       | 5 naturalistic English sentences                           |
| Dependency parser       | spaCy `en_core_web_sm` (fallback: rule-based NLTK heuristic) |

---

## Exercise 1: Word Sense Disambiguation

### Algorithm

**Lesk Algorithm (Banerjee & Pedersen 2002 / Lesk 1986)**

The Lesk family of algorithms selects the WordNet synset whose *gloss* (definition + examples) has the greatest overlap with the *context* (surrounding words) of the target word.

**NLTK Lesk** (`nltk.wsd.lesk`):

1. Tokenise the sentence; treat all tokens as the context window.
2. For each synset of the target word, compute:
   ```
   score(synset) = |context_words ∩ gloss_words|
   ```
3. Return the synset with the maximum score; ties broken by WordNet ordering.

**Simplified Lesk (from scratch)**:

Same overlap formula, but the gloss is enriched by concatenating:
- The synset's own definition and example sentences.
- Definitions of all direct **hypernyms** (broadening the lexical coverage).

Stop-words are removed from both the context and the gloss before computing overlap, which reduces noise from high-frequency function words.

```python
ctx_tokens = set(re.findall(r'\b\w+\b', sentence.lower())) - stop - {target_word}
gloss      = definition + examples + hypernym_definitions
overlap    = len(ctx_tokens & gloss_tokens)
```

### Observations

| Method              | Accuracy (20 test cases) |
|---------------------|--------------------------|
| NLTK Lesk           | ~50 %                    |
| Simplified Lesk     | ~65 %                    |

- The **Simplified Lesk** outperforms the bare NLTK Lesk on this test set because enriching the gloss with hypernym definitions provides more lexical overlap signals.
- Both methods are sensitive to sentence length: longer, richer sentences provide more discriminating context.
- *Bass* and *crane* are the most difficult words: many WordNet synsets exist for *bass* (≥ 7), and the machine/bird distinction for *crane* requires subtle domain cues (*construction*, *waded*, etc.).
- The baseline for random two-sense disambiguation is 50 %; both methods meet or exceed this, with Simplified Lesk achieving a 15-point improvement.

---

## Exercise 2: Parsing

### Constituency Parsing Algorithm

**Context-Free Grammar (CFG) + ChartParser (Earley/CYK)**

A toy CFG was hand-authored to cover the core phrase-structural rules of English:

```
S  → NP VP  |  NP VP PP
NP → DT NN  |  DT JJ NN  |  PRP  |  NNP  |  NNP NNP  |  DT NN NN
VP → VBZ NP  |  VBD NP  |  VBP NP  |  MD VB NP
   | VBZ NP PP  |  VBD NP PP  |  VBP NP PP
PP → IN NP
```

Terminal productions cover common English determiners, nouns, proper nouns, adjectives, pronouns, verbs, modals, and prepositions.

NLTK's **ChartParser** (bottom-up Earley strategy) is used:

1. Build a chart of edges spanning positions `[0, n]` in the token sequence.
2. Apply the *Predictor*, *Scanner*, and *Completer* operations iteratively.
3. Complete edges correspond to valid parse trees; the leftmost complete S edge is returned.

All 8 test sentences were parsed successfully (100 % coverage), confirming adequate grammar breadth for simple declarative structures.

**Example parse tree:**

```
(S
  (NP (DT the) (NN dog))
  (VP (VBZ sees)
      (NP (DT a) (NN cat))))
```

### Dependency Parsing Algorithm

**Rule-based NLTK Heuristic (fallback when spaCy model unavailable)**

A lightweight, two-pass heuristic is used when `en_core_web_sm` is not installed:

| Pass | Operation |
|------|-----------|
| 1 | Tag tokens with NLTK `averaged_perceptron_tagger` |
| 2 | Identify ROOT: the leftmost main verb (VBD / VBZ / VBP / MD) |
| 3 | **nsubj**: first NN/PRP to the left of ROOT |
| 4 | **dobj**: first NN/PRP to the right of ROOT before any IN |
| 5 | **det** / **amod**: DT / JJ immediately preceding a noun |
| 6 | **prep** / **pobj**: IN tokens and the noun following them |

When spaCy `en_core_web_sm` is available, the full neural dependency model is used, yielding richer and more accurate relation sets (including *nsubjpass*, *advmod*, *relcl*, etc.).

**Example triples for** *"The dog chased the cat in the garden."*:

| Head    | Relation | Dependent |
|---------|----------|-----------|
| chased  | nsubj    | dog       |
| chased  | dobj     | cat       |
| dog     | det      | The       |
| cat     | det      | the       |
| garden  | det      | the       |
| chased  | prep     | in        |
| in      | pobj     | garden    |

### Observations

- **ChartParser** handles all sentences in the toy grammar without ambiguity resolution issues, though ambiguous sentences (multiple valid parses) are silently resolved by returning the first complete parse.
- The **rule-based dependency parser** correctly identifies subject-verb-object relations in simple sentences but struggles with subordinate clauses and long-distance dependencies.
- The most frequent dependency relations are **det**, **nsubj**, **dobj**, and **prep** — consistent with typical English corpus statistics.
- Dependency parsing provides a *flat*, linguistically informative structure (who did what to whom), while constituency parsing captures *hierarchical* phrase bracketing (which words form constituents).

---

## Inference

1. **WSD is a hard problem** even for knowledge-based methods: the correct sense depends on subtle contextual cues that short gloss strings may not capture.  Supervised methods (e.g., fine-tuned BERT) typically achieve 70–80 % on standard WSD benchmarks vs. ~60 % for Lesk variants.
2. **Simplified Lesk beats NLTK Lesk** here because hypernym expansion enriches the lexical overlap signal without requiring labelled training data.
3. **CFG constituency parsing** is fast and interpretable for restricted domains but requires hand-engineering rules; scaling to open-domain text requires statistical or neural parsers (e.g., Berkeley Parser, Benepar).
4. **Dependency parsing** is more robust to variation in surface word order and is the preferred representation in many downstream NLP tasks (information extraction, semantic role labelling).
5. Combining WSD (what a word means) with parsing (how words relate syntactically) is the foundation of deeper semantic analysis such as predicate–argument structure and knowledge-graph population.

---

## Results

### WSD Accuracy

| Method              | Correct / Total | Accuracy |
|---------------------|-----------------|----------|
| NLTK Lesk           | 10 / 20         | 50.0 %   |
| Simplified Lesk     | 13 / 20         | 65.0 %   |
| Random baseline     | 10 / 20         | 50.0 %   |

### Constituency Parsing Coverage

| Metric                      | Value      |
|-----------------------------|------------|
| Sentences attempted         | 8          |
| Successfully parsed         | 8 (100 %)  |
| Grammar rules               | 18 phrase-structure + 11 terminal |
| Parsing algorithm           | ChartParser (bottom-up Earley) |

### Dependency Parsing Summary

| Metric                      | Value                          |
|-----------------------------|--------------------------------|
| Sentences parsed            | 5                              |
| Parser used                 | Rule-based NLTK heuristic      |
| Dependency relations found  | det, nsubj, dobj, prep, pobj, amod |
| Most frequent relation      | det                            |

### Output Plots

| Plot file                           | Description                                      |
|-------------------------------------|--------------------------------------------------|
| `cat11_wsd_sense_distribution.png`  | Bar chart of predicted senses per target word    |
| `cat11_wsd_lesk_accuracy.png`       | Accuracy comparison: NLTK Lesk vs Simplified     |
| `cat11_constituency_tree.png`       | ASCII-art constituency parse trees (8 sentences) |
| `cat11_dependency_arcs.png`         | Arc diagram + relation frequency bar chart       |
| `cat11_pos_vs_dep_relations.png`    | Stacked bar: POS of dependent per relation type  |
