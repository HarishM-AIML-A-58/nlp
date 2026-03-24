# Lab Record — Category 1: Text Preprocessing

## Aim
To implement a complete text preprocessing pipeline for Natural Language Processing tasks.

## Objective
1. Apply text normalization (lowercasing, unicode, punctuation removal)
2. Perform word tokenization and sentence segmentation
3. Remove stopwords to reduce noise
4. Apply stemming (Porter, Snowball) and lemmatization (NLTK WordNet, spaCy)
5. Compare corpus statistics at each pipeline stage

## Dataset Description
- **Brown Corpus** (NLTK): ~1 million words, 500 sentences sampled, balanced across 15 genres (news, fiction, government, etc.)
- **Reuters Corpus** (NLTK): News wire corpus, 500 sentences sampled from ~10,000 documents

## Algorithm

### Stepwise Pipeline
1. **Input**: Raw text string
2. **HTML/URL removal**: Strip `<tag>` and `http://...` patterns
3. **Lowercasing**: Convert all characters to lowercase
4. **Punctuation removal**: Remove all punctuation characters
5. **Whitespace normalization**: Collapse multiple spaces to single space
6. **Tokenization**: Split text into word tokens using NLTK punkt tokenizer
7. **Stopword removal**: Filter tokens matching NLTK English stopword list (179 words)
8. **Stemming**: Apply Porter stemmer to reduce inflected forms to stem
9. **Lemmatization**: Apply WordNet lemmatizer to convert to dictionary base form

### Key Differences
| Operation | Input | Output | Example |
|-----------|-------|--------|---------|
| Stemming  | "running" | "run" | Algorithmic, may not be a real word |
| Lemmatization | "better" | "good" | Dictionary-based, always valid word |

## Observations
- **Token reduction**: Stopword removal reduces token count by ~40–50% on average
- **Vocabulary reduction**: Lemmatization reduces vocabulary by ~15–20% compared to raw tokens
- **Top words shift**: Raw text dominated by function words ("the", "a", "of"); after stopword removal, content words emerge ("government", "said", "year")
- **Stemming aggressiveness**: Porter stemmer strips more than lemmatization ("organization" → "organ" vs "organization")
- **Reuters vs Brown**: Reuters has higher type-token ratio (more diverse vocabulary) due to specialized news content

## Inference
- Text preprocessing significantly reduces noise and dimensionality, critical for downstream NLP tasks
- Lemmatization is preferred over stemming when interpretability matters
- Stopword removal is domain-dependent — "not" should be retained for sentiment analysis
- The Brown corpus has more uniform word distribution; Reuters has heavier tails (Zipfian)

## Results

| Stage | Brown Tokens | Brown Vocab | Reuters Tokens | Reuters Vocab |
|-------|-------------|-------------|----------------|---------------|
| Raw | ~12,000 | ~3,500 | ~11,000 | ~3,800 |
| Cleaned | ~11,500 | ~3,200 | ~10,500 | ~3,500 |
| No Stopwords | ~7,000 | ~3,100 | ~6,500 | ~3,400 |
| Stemmed | ~7,000 | ~2,600 | ~6,500 | ~2,800 |
| Lemmatized | ~7,000 | ~2,900 | ~6,500 | ~3,100 |

**Outputs**: `outputs/plots/cat1_*.png`, `outputs/metrics/cat1_preprocessing.json`
