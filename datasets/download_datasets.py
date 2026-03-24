#!/usr/bin/env python3
"""
Dataset download script for NLP Lab Repository.
Downloads and verifies all required datasets.
"""

import os
import sys

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_nltk_corpora():
    """Download all required NLTK corpora and resources."""
    import nltk
    resources = [
        ('corpora', 'brown'),
        ('corpora', 'reuters'),
        ('corpora', 'wordnet'),
        ('corpora', 'stopwords'),
        ('corpora', 'omw-1.4'),
        ('corpora', 'words'),
        ('tokenizers', 'punkt'),
        ('tokenizers', 'punkt_tab'),
        ('taggers', 'averaged_perceptron_tagger'),
        ('taggers', 'averaged_perceptron_tagger_eng'),
    ]
    print("[NLTK Downloads]")
    for category, name in resources:
        try:
            nltk.data.find(f'{category}/{name}')
            print(f"  ✓ {name} (already present)")
        except LookupError:
            print(f"  ↓ Downloading {name}...")
            nltk.download(name, quiet=False)


def download_newsgroups():
    """Download 20 Newsgroups via sklearn."""
    print("\n[20 Newsgroups]")
    try:
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='all', download_if_missing=True)
        print(f"  ✓ 20 Newsgroups: {len(data.data)} documents, {len(data.target_names)} categories")

        # Save category list
        cats_file = os.path.join(DATASETS_DIR, 'newsgroups', 'categories.txt')
        os.makedirs(os.path.dirname(cats_file), exist_ok=True)
        with open(cats_file, 'w') as f:
            f.write('\n'.join(data.target_names))
        print(f"  Saved category list to datasets/newsgroups/categories.txt")
    except Exception as e:
        print(f"  ! Could not download 20 Newsgroups: {e}")


def download_ud_treebank():
    """Download Universal Dependencies English Treebank via NLTK or conllu."""
    print("\n[Universal Dependencies Treebank]")
    ud_dir = os.path.join(DATASETS_DIR, 'ud_treebank')
    os.makedirs(ud_dir, exist_ok=True)

    # Try via NLTK's conllu if available
    try:
        import nltk
        nltk.download('dependency_treebank', quiet=True)
        print("  ✓ Dependency treebank (NLTK built-in)")
    except Exception:
        pass

    # Write a small sample UD file for offline use
    sample_conllu = """# sent_id = 1
# text = The dog runs fast.
1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_
2\tdog\tdog\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_
3\truns\trun\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\t_
4\tfast\tfast\tADV\tRB\t_\t3\tadvmod\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t.\tPunctType=Peri\t3\tpunct\t_\t_

# sent_id = 2
# text = A cat sleeps quietly.
1\tA\ta\tDET\tDT\tDefinite=Ind|PronType=Art\t2\tdet\t_\t_
2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_
3\tsleeps\tsleep\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\t_
4\tquietly\tquietly\tADV\tRB\t_\t3\tadvmod\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t.\tPunctType=Peri\t3\tpunct\t_\t_
"""
    sample_file = os.path.join(ud_dir, 'sample.conllu')
    with open(sample_file, 'w') as f:
        f.write(sample_conllu)
    print(f"  ✓ Sample UD CoNLL-U file saved to datasets/ud_treebank/sample.conllu")


def verify_large_corpus():
    """Check or create a placeholder for large corpus."""
    print("\n[Large Corpus]")
    large_dir = os.path.join(DATASETS_DIR, 'large_corpus')
    os.makedirs(large_dir, exist_ok=True)
    readme = os.path.join(large_dir, 'README.txt')

    with open(readme, 'w') as f:
        f.write("""Large Corpus Directory
======================
Place large text corpora here for extended experiments.

Recommended options:
1. Wikipedia dump (English):
   wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

2. Common Crawl subset (via datasets library):
   from datasets import load_dataset
   ds = load_dataset('cc_news', split='train', streaming=True)

3. OpenWebText (via datasets library):
   from datasets import load_dataset
   ds = load_dataset('openwebtext', split='train', streaming=True)

For the lab experiments, the Brown + Reuters corpora are sufficient.
The experiments/cat9_word_embeddings/ scripts automatically use Brown + Reuters.
""")
    print(f"  ✓ README saved to datasets/large_corpus/README.txt")


def save_corpus_stats():
    """Save corpus statistics to a JSON file."""
    import json
    import nltk
    from nltk.corpus import brown, reuters

    stats = {}

    try:
        stats['brown'] = {
            'num_words': len(brown.words()),
            'num_sentences': len(brown.sents()),
            'num_categories': len(brown.categories()),
            'categories': brown.categories(),
        }
        print(f"\n  Brown corpus: {stats['brown']['num_words']:,} words")
    except Exception as e:
        stats['brown'] = {'error': str(e)}

    try:
        stats['reuters'] = {
            'num_words': len(reuters.words()),
            'num_sentences': len(reuters.sents()),
            'num_categories': len(reuters.categories()),
        }
        print(f"  Reuters corpus: {stats['reuters']['num_words']:,} words")
    except Exception as e:
        stats['reuters'] = {'error': str(e)}

    stats_file = os.path.join(DATASETS_DIR, 'corpus_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  Corpus stats saved to datasets/corpus_stats.json")


def main():
    print("=" * 60)
    print("NLP LAB — DATASET DOWNLOAD SCRIPT")
    print("=" * 60)

    download_nltk_corpora()
    download_newsgroups()
    download_ud_treebank()
    verify_large_corpus()
    save_corpus_stats()

    print("\n" + "=" * 60)
    print("Dataset download complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
