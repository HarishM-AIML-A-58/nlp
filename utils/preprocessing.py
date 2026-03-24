"""
Preprocessing utilities for NLP lab experiments.
Provides normalization, cleaning, tokenization, stopword removal,
stemming, and lemmatization functions.
"""

import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download required NLTK resources
_NLTK_RESOURCES = [
    'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
    'punkt_tab', 'omw-1.4'
]

def ensure_nltk_resources():
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

ensure_nltk_resources()

# Lazy-load spaCy model
_nlp = None

def get_spacy_model():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
            _nlp = spacy.load('en_core_web_sm')
    return _nlp


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def remove_special_characters(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


# ─── Full Cleaning Pipeline ───────────────────────────────────────────────────

def clean_text(text: str,
               do_lowercase: bool = True,
               do_remove_urls: bool = True,
               do_remove_html: bool = True,
               do_remove_numbers: bool = False,
               do_remove_punctuation: bool = True,
               do_remove_extra_whitespace: bool = True) -> str:
    """Full cleaning pipeline."""
    if do_remove_html:
        text = remove_html_tags(text)
    if do_remove_urls:
        text = remove_urls(text)
    if do_lowercase:
        text = lowercase(text)
    if do_remove_numbers:
        text = remove_numbers(text)
    if do_remove_punctuation:
        text = remove_punctuation(text)
    if do_remove_extra_whitespace:
        text = remove_extra_whitespace(text)
    return text


# ─── Tokenization ─────────────────────────────────────────────────────────────

def word_tokenize_nltk(text: str) -> list:
    """Tokenize text into words using NLTK."""
    return word_tokenize(text)


def sent_tokenize_nltk(text: str) -> list:
    """Tokenize text into sentences using NLTK."""
    return sent_tokenize(text)


def simple_tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return text.split()


# ─── Stopword Removal ─────────────────────────────────────────────────────────

def get_stopwords(language: str = 'english') -> set:
    return set(stopwords.words(language))


def remove_stopwords(tokens: list, language: str = 'english') -> list:
    sw = get_stopwords(language)
    return [t for t in tokens if t.lower() not in sw]


# ─── Stemming ─────────────────────────────────────────────────────────────────

def stem_porter(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def stem_snowball(tokens: list, language: str = 'english') -> list:
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(t) for t in tokens]


# ─── Lemmatization ───────────────────────────────────────────────────────────

def lemmatize_nltk(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def lemmatize_spacy(tokens: list) -> list:
    nlp = get_spacy_model()
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def full_pipeline(text: str,
                  remove_stops: bool = True,
                  stemming: bool = False,
                  lemmatization: bool = True) -> dict:
    """
    Run full preprocessing pipeline and return each stage.
    """
    original = text
    cleaned = clean_text(text)
    tokens = word_tokenize_nltk(cleaned)

    no_stops = remove_stopwords(tokens) if remove_stops else tokens

    stemmed = stem_porter(no_stops) if stemming else None
    lemmatized = lemmatize_nltk(no_stops) if lemmatization else None

    return {
        'original': original,
        'cleaned': cleaned,
        'tokens': tokens,
        'tokens_no_stopwords': no_stops,
        'stemmed': stemmed,
        'lemmatized': lemmatized,
    }
