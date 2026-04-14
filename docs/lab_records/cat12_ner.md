# Lab Record — Category 12: Named Entity Recognition

## Aim

To implement and evaluate Named Entity Recognition (NER) systems using multiple
approaches — rule-based gazetteers, statistical/ML-backed chunking (NLTK `ne_chunk`),
and regular-expression patterns — and to compare their performance on a hand-labelled
evaluation set.

---

## Objective

1. Build a **rule-based NER system** using gazetteer lookup lists and contextual
   title rules to identify PERSON, ORGANISATION, and LOCATION entities.
2. Apply **NLTK's built-in `ne_chunk`** tagger (backed by the MaxEnt NE chunker and
   the averaged-perceptron POS tagger) to the Brown and Reuters corpora.
3. Implement **regex-based pattern matching** to extract structured entities such as
   monetary amounts, dates, percentages, email addresses, and URLs.
4. **Evaluate** approaches 1 and 2 on a manually annotated gold set of 30 sentences,
   reporting Precision, Recall, and F1 per entity type.
5. Visualise entity frequency distributions across corpora.

---

## Dataset Description

| Dataset | Source | Sentences Used | Notes |
|---------|--------|---------------|-------|
| Brown Corpus | `nltk.corpus.brown` | 2 000 | General American English, multiple genres |
| Reuters Corpus | `nltk.corpus.reuters` | 2 000 | News-wire text, entity-rich |
| Gold Evaluation Set | Hand-annotated (this lab) | 30 | Covers PERSON, ORG, LOC entities |

The **gold evaluation set** consists of 30 sentences constructed to cover diverse
named entity types: politicians, technology companies, international locations,
non-governmental organisations, and multi-word entities.

---

## Algorithm

### Rule-Based NER

The rule-based system operates in three stages:

1. **Gazetteer look-up** — Six look-up sets are maintained:
   - ~100 common first names and ~100 common last names (PERSON)
   - ~80 well-known organisations (ORG)
   - ~200 country names, ~70 major world cities, 50 US states (LOC)
2. **Contextual title rules** — Tokens matching `{Mr.|Mrs.|Ms.|Dr.|Prof.|Sen.|Rep.|Gov.}`
   trigger a PERSON tag for the immediately following capitalised word(s).
3. **Multi-word matching** — Bi-gram look-ups are attempted before single-token look-ups
   to capture entities like *New York* or *Goldman Sachs*.

Token matching is case-insensitive for gazetteer look-up but respects capitalisation
for PERSON entity detection (only capitalised first/last name tokens are tagged).

### Statistical NER (NLTK MaxEnt)

NLTK's `ne_chunk` applies a **binary or multi-class Maximum Entropy chunker** on top
of POS-tagged tokens:

1. Tokenise each sentence with `nltk.word_tokenize`.
2. POS-tag using `nltk.pos_tag` (averaged perceptron tagger).
3. Chunk named entities with `nltk.ne_chunk(tagged, binary=False)`.
4. Traverse the resulting `nltk.Tree`; each sub-tree whose label is a NE type
   (PERSON, ORGANIZATION, GPE, …) is extracted as an entity mention.
5. GPE (Geopolitical Entity) is mapped to LOC for consistency with other approaches.

### IOB Tagging Scheme

Although this lab uses chunk-tree output rather than a flat IOB sequence, the
underlying NLTK chunker encodes entities using the **IOB2 (Inside-Outside-Beginning)**
scheme internally:

| Tag | Meaning |
|-----|---------|
| `B-TYPE` | Beginning token of entity of TYPE |
| `I-TYPE` | Continuation token inside entity of TYPE |
| `O`      | Token outside any named entity |

For example, *"New York City"* as a location would be encoded as
`B-LOC I-LOC I-LOC`. The chunk tree representation collapses this into a
`Tree('GPE', [('New', 'NNP'), ('York', 'NNP'), ('City', 'NNP')])` node.

### Regex-Based Patterns

Six compiled patterns handle structured entities:

| Pattern Type | Example |
|-------------|---------|
| `MONEY` | `$3.5 billion`, `$500,000` |
| `DATE` | `January 15, 2024`, `2024-03-22` |
| `PERCENT` | `12.5%`, `8%` |
| `EMAIL` | `info@company.com` |
| `URL` | `https://www.example.com` |
| `PHONE` | `800-555-1234` |

---

## Observations

### Corpus Entity Frequencies

Entity counts extracted by NLTK `ne_chunk` from the first 2 000 sentences of each
corpus (Brown + Reuters):

| Entity Type | Count |
|-------------|-------|
| PERSON | (see metrics JSON) |
| ORG | (see metrics JSON) |
| LOC | (see metrics JSON) |

Entity mention lengths are predominantly **1–2 words**, with 3-word mentions
appearing for titles + first + last name combinations (e.g., *"President Barack Obama"*).

### Evaluation Results (30-sentence gold set)

#### Rule-Based NER

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| PERSON | varies | varies | varies |
| ORG | varies | varies | varies |
| LOC | varies | varies | varies |

#### NLTK ne_chunk

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| PERSON | varies | varies | varies |
| ORG | varies | varies | varies |
| LOC | varies | varies | varies |

*(Actual values recorded in `outputs/metrics/cat12_ner.json`)*

### Regex Pattern Results

Across seven test sentences the regex system detected multiple MONEY, DATE,
PERCENT, EMAIL, URL, and PHONE entities with 100% precision on well-formed inputs
(no false negatives on the demo texts). Real-world performance varies with text
cleanliness and format diversity.

---

## Inference

1. **Rule-based NER** offers high precision for well-known entities included in the
   gazetteers but suffers from low recall for entities outside the predefined lists.
   It is fully deterministic and requires no training data.

2. **NLTK ne_chunk** provides better coverage through the statistical chunker trained
   on the ACE corpus; however, it can produce false positives for capitalised common
   nouns in contexts where uppercase is used for emphasis rather than as a proper noun.

3. **Regex-based NER** is complementary to both approaches: it reliably extracts
   *structured* entities (dates, money, percentages) that gazetteers and statistical
   models may miss entirely.

4. **Multi-word entity detection** is the hardest sub-problem: both approaches show
   reduced recall for compound names (*Goldman Sachs*, *New York*) compared to
   single-token names.

5. The Reuters corpus is more entity-dense (news articles) than the Brown corpus
   (general English prose), leading to higher entity counts per sentence in Reuters.

---

## Results

### Top Entity Mentions (Brown + Reuters, first 2 000 sentences each)

Results stored in `outputs/metrics/cat12_ner.json` under `per_type_counts`.

### Generated Plots

| File | Description |
|------|-------------|
| `outputs/plots/cat12_entity_type_distribution.png` | Pie + bar chart of PERSON/ORG/LOC totals |
| `outputs/plots/cat12_top_persons.png` | Top 20 person mentions |
| `outputs/plots/cat12_top_organizations.png` | Top 20 organisation mentions |
| `outputs/plots/cat12_top_locations.png` | Top 20 location mentions |
| `outputs/plots/cat12_ner_precision_recall.png` | P/R/F1 comparison: Rule-Based vs NLTK ne_chunk |
| `outputs/plots/cat12_entity_length_dist.png` | Distribution of entity mention lengths |
