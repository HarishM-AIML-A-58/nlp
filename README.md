# NLP Lab Repository

A complete, production-grade Natural Language Processing (NLP) lab repository covering core NLP fundamentals, statistical language modeling, morphological analysis, neural language models, and word embeddings.

## Repository Structure

```
nlp/
├── experiments/
│   ├── cat1_preprocessing/       # Text Preprocessing
│   ├── cat2_entropy_perplexity/  # Entropy, Cross-Entropy, Perplexity
│   ├── cat3_rule_morphology/     # Rule-Based Morphological Analyzer
│   ├── cat4_fsa_morphology/      # FSA-Based Morphological Analyzer
│   ├── cat5_unigram_model/       # Unigram Language Model
│   ├── cat6_bigram_model/        # Bigram Language Model
│   ├── cat7_neural_models/       # Neural Language Models
│   ├── cat8_vector_semantics/    # Vector Semantics
│   └── cat9_word_embeddings/     # Word Embeddings + Visualization
├── datasets/                     # Corpus data
├── utils/                        # Reusable modules
├── outputs/plots/                # Generated visualizations
├── outputs/metrics/              # Computed metrics
├── docs/lab_records/             # Lab documentation (Word files)
├── notebooks/                    # Jupyter notebooks
├── requirements.txt
├── run_all.py                    # Master execution script
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download NLTK data

```bash
python -c "
import nltk
nltk.download('brown')
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
"
```

## Running Experiments

### Run all experiments

```bash
python run_all.py
```

### Run individual categories

```bash
python experiments/cat1_preprocessing/run.py
python experiments/cat2_entropy_perplexity/run.py
python experiments/cat3_rule_morphology/run.py
python experiments/cat4_fsa_morphology/run.py
python experiments/cat5_unigram_model/run.py
python experiments/cat6_bigram_model/run.py
python experiments/cat7_neural_models/run.py
python experiments/cat8_vector_semantics/run.py
python experiments/cat9_word_embeddings/run.py
```

## Categories Overview

| Category | Topic | Key Techniques |
|----------|-------|----------------|
| 1 | Text Preprocessing | Tokenization, Stemming, Lemmatization |
| 2 | Entropy & Perplexity | Shannon Entropy, Cross-Entropy, Perplexity |
| 3 | Rule-Based Morphology | Prefix/Suffix rules, POS transformations |
| 4 | FSA Morphology | Finite State Automata |
| 5 | Unigram LM | Frequency-based probabilities, Laplace smoothing |
| 6 | Bigram LM | Bigram probabilities, Good-Turing, Kneser-Ney |
| 7 | Neural LM | Spelling correction, CBOW, Skip-gram |
| 8 | Vector Semantics | TF-IDF, Cosine similarity, PMI |
| 9 | Word Embeddings | Word2Vec, PCA, t-SNE, UMAP |

## Datasets Used

- **Brown Corpus** (NLTK) — 1M+ words, balanced genres
- **Reuters Corpus** (NLTK) — news articles
- **20 Newsgroups** (sklearn) — 20,000 documents
- **Universal Dependencies Treebank** — syntactic annotations
- **Large Corpus** — Wikipedia/Common Crawl subset

## Outputs

All outputs are saved to `outputs/`:
- `plots/` — Visualizations (PNG)
- `metrics/` — Computed metrics (JSON/CSV)

## Lab Records

Documentation for each category is in `docs/lab_records/` including:
- Aim, Objective
- Dataset description
- Algorithm (stepwise)
- Observations & Inferences
- Results

## Authors

Built for academic and portfolio purposes demonstrating NLP fundamentals.
