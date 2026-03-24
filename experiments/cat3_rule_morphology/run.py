"""
Category 3: Rule-Based Morphological Analyzer
==============================================
Aim: Build a rule-based system for morphological analysis of English words.
Objective: Implement prefix/suffix stripping rules, identify morphemes,
           and perform POS-based word transformations.
Dataset: Brown Corpus (NLTK)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import brown

from utils.evaluation import save_metrics

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# ─── Morphological Rule Database ─────────────────────────────────────────────

# Suffix rules: (suffix, replacement, pos_result, description)
SUFFIX_RULES = [
    # Verb → Noun
    ('tion',  '',    'NOUN',  'nominalisation'),
    ('ation', '',    'NOUN',  'nominalisation'),
    ('ment',  '',    'NOUN',  'nominalisation'),
    ('ness',  '',    'NOUN',  'state/quality'),
    ('ity',   '',    'NOUN',  'state/quality'),
    ('ance',  '',    'NOUN',  'state'),
    ('ence',  '',    'NOUN',  'state'),
    ('er',    '',    'NOUN',  'agent'),
    ('or',    '',    'NOUN',  'agent'),
    ('ist',   '',    'NOUN',  'person'),
    ('ism',   '',    'NOUN',  'doctrine'),
    # Adjective
    ('ful',   '',    'ADJ',   'full of'),
    ('less',  '',    'ADJ',   'without'),
    ('ous',   '',    'ADJ',   'having quality'),
    ('ious',  '',    'ADJ',   'having quality'),
    ('al',    '',    'ADJ',   'pertaining to'),
    ('ical',  '',    'ADJ',   'pertaining to'),
    ('ive',   '',    'ADJ',   'tending to'),
    ('able',  '',    'ADJ',   'capable of'),
    ('ible',  '',    'ADJ',   'capable of'),
    # Verb
    ('ing',   '',    'VERB',  'progressive'),
    ('ed',    '',    'VERB',  'past tense'),
    ('ize',   '',    'VERB',  'to make'),
    ('ise',   '',    'VERB',  'to make'),
    ('fy',    '',    'VERB',  'to make'),
    # Adverb
    ('ly',    '',    'ADV',   'manner'),
]

# Prefix rules: (prefix, description)
PREFIX_RULES = [
    ('un',  'negation'),
    ('dis', 'negation'),
    ('in',  'negation'),
    ('im',  'negation'),
    ('non', 'negation'),
    ('anti','against'),
    ('pre', 'before'),
    ('post','after'),
    ('re',  'again'),
    ('over','excess'),
    ('under','insufficient'),
    ('sub', 'below'),
    ('super','above'),
    ('co',  'together'),
    ('inter','between'),
    ('trans','across'),
    ('micro','small'),
    ('macro','large'),
    ('auto', 'self'),
    ('bio',  'life'),
]

# ─── Rule-Based Morphological Analyzer ───────────────────────────────────────

class RuleBasedMorphAnalyzer:
    """
    Analyzes words by detecting prefixes, suffixes, and stems
    using hand-crafted morphological rules.
    """

    MIN_STEM_LENGTH = 3  # Minimum stem length after stripping

    def __init__(self):
        # Sort by suffix length descending (longest match first)
        self.suffix_rules = sorted(SUFFIX_RULES, key=lambda x: -len(x[0]))
        self.prefix_rules = sorted(PREFIX_RULES, key=lambda x: -len(x[0]))

    def analyze_suffixes(self, word: str) -> list:
        """Return list of (suffix, stem, pos, description) matches."""
        word_lower = word.lower()
        results = []
        for suffix, replacement, pos, desc in self.suffix_rules:
            if word_lower.endswith(suffix):
                stem = word_lower[:-len(suffix)] + replacement
                if len(stem) >= self.MIN_STEM_LENGTH:
                    results.append({
                        'word': word,
                        'suffix': suffix,
                        'stem': stem,
                        'pos': pos,
                        'description': desc,
                    })
        return results

    def analyze_prefixes(self, word: str) -> list:
        word_lower = word.lower()
        results = []
        for prefix, desc in self.prefix_rules:
            if word_lower.startswith(prefix):
                root = word_lower[len(prefix):]
                if len(root) >= self.MIN_STEM_LENGTH:
                    results.append({
                        'word': word,
                        'prefix': prefix,
                        'root': root,
                        'description': desc,
                    })
        return results

    def analyze(self, word: str) -> dict:
        """Full morphological analysis of a single word."""
        prefixes = self.analyze_prefixes(word)
        suffixes = self.analyze_suffixes(word)

        # Best guess POS
        pos_guess = 'UNKNOWN'
        if suffixes:
            pos_guess = suffixes[0]['pos']

        return {
            'word': word,
            'length': len(word),
            'prefixes': prefixes,
            'suffixes': suffixes,
            'pos_guess': pos_guess,
            'num_morphemes': len(prefixes) + len(suffixes) + 1,
        }

    def batch_analyze(self, words: list) -> list:
        return [self.analyze(w) for w in words]


# ─── POS-Based Transformations ────────────────────────────────────────────────

TRANSFORMATION_RULES = [
    # (pattern, pos, transform, result_pos, example)
    (r'(\w+)ful$',  'ADJ', lambda m: m.group(1),          'NOUN', 'helpful→help'),
    (r'(\w+)ness$', 'NOUN', lambda m: m.group(1),         'ADJ',  'happiness→happy'),
    (r'(\w+)ly$',   'ADV', lambda m: m.group(1),          'ADJ',  'quickly→quick'),
    (r'(\w+)ize$',  'VERB', lambda m: m.group(1) + 'ism', 'NOUN', 'organize→organism'),
    (r'(\w+)ed$',   'VERB', lambda m: m.group(1),         'VERB', 'walked→walk'),
    (r'(\w+)ing$',  'VERB', lambda m: m.group(1),         'VERB', 'running→run'),
    (r'(\w+)er$',   'NOUN', lambda m: m.group(1),         'VERB', 'teacher→teach'),
    (r'(\w+)tion$', 'NOUN', lambda m: m.group(1) + 'te',  'VERB', 'creation→create'),
]

class POSTransformer:
    def transform(self, word: str) -> list:
        results = []
        for pattern, src_pos, transform_fn, result_pos, example in TRANSFORMATION_RULES:
            match = re.match(pattern, word.lower())
            if match:
                try:
                    derived = transform_fn(match)
                    results.append({
                        'input': word,
                        'src_pos': src_pos,
                        'derived': derived,
                        'result_pos': result_pos,
                        'rule_example': example,
                    })
                except Exception:
                    pass
        return results

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_on_corpus(analyzer: RuleBasedMorphAnalyzer, corpus_sents: list, n: int = 2000):
    """Analyze words from corpus and collect statistics."""
    words = set()
    for sent in corpus_sents[:n]:
        for word in sent:
            if word.isalpha() and len(word) > 3:
                words.add(word.lower())

    results = analyzer.batch_analyze(list(words)[:1000])

    pos_counts    = collections.Counter(r['pos_guess'] for r in results)
    suffix_counts = collections.Counter(
        s['suffix'] for r in results for s in r['suffixes']
    )
    prefix_counts = collections.Counter(
        p['prefix'] for r in results for p in r['prefixes']
    )
    morpheme_dist = collections.Counter(r['num_morphemes'] for r in results)

    return {
        'total_words': len(results),
        'pos_distribution': dict(pos_counts),
        'top_suffixes': dict(suffix_counts.most_common(10)),
        'top_prefixes': dict(prefix_counts.most_common(10)),
        'morpheme_count_dist': {str(k): v for k, v in morpheme_dist.items()},
        'words_with_prefix': sum(1 for r in results if r['prefixes']),
        'words_with_suffix': sum(1 for r in results if r['suffixes']),
    }

# ─── Visualization ────────────────────────────────────────────────────────────

def plot_pos_distribution(pos_dist: dict):
    labels = [k for k, v in sorted(pos_dist.items(), key=lambda x: -x[1])]
    counts = [pos_dist[k] for k in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, counts, color=sns.color_palette('Set2', len(labels)))
    ax.set_title('POS Distribution (Rule-Based Morphology)')
    ax.set_ylabel('Word Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat3_pos_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: cat3_pos_distribution.png")


def plot_suffix_counts(suffix_counts: dict):
    items = sorted(suffix_counts.items(), key=lambda x: -x[1])[:15]
    labels, counts = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(labels)[::-1], list(counts)[::-1], color='steelblue')
    ax.set_title('Most Common Suffixes Detected')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat3_suffix_counts.png'), dpi=150)
    plt.close()
    print("  Saved: cat3_suffix_counts.png")


def plot_prefix_counts(prefix_counts: dict):
    if not prefix_counts:
        return
    items = sorted(prefix_counts.items(), key=lambda x: -x[1])[:15]
    labels, counts = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(labels)[::-1], list(counts)[::-1], color='coral')
    ax.set_title('Most Common Prefixes Detected')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat3_prefix_counts.png'), dpi=150)
    plt.close()
    print("  Saved: cat3_prefix_counts.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 3: RULE-BASED MORPHOLOGICAL ANALYZER")
    print("=" * 60)

    nltk.download('brown', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    analyzer    = RuleBasedMorphAnalyzer()
    transformer = POSTransformer()

    # Demo on curated word list
    demo_words = [
        'unhappiness', 'predisposition', 'internationalization',
        'reprocessing', 'comfortable', 'quickly', 'teachers',
        'organization', 'underdevelopment', 'microbiological',
        'autoimmune', 'preprocessing', 'transformation',
    ]
    print("\n[Demo: Morphological Analysis]")
    for word in demo_words:
        result = analyzer.analyze(word)
        prefix_str = ', '.join(p['prefix'] for p in result['prefixes']) or '-'
        suffix_str = ', '.join(s['suffix'] for s in result['suffixes']) or '-'
        print(f"  {word:<25} | POS: {result['pos_guess']:<8} | "
              f"prefixes: {prefix_str:<10} | suffixes: {suffix_str}")

    # POS transformations
    print("\n[Demo: POS-Based Transformations]")
    transform_words = ['helpful', 'happiness', 'quickly', 'walked', 'running']
    for word in transform_words:
        transforms = transformer.transform(word)
        for t in transforms:
            print(f"  {word} ({t['src_pos']}) → {t['derived']} ({t['result_pos']})")

    # Corpus evaluation
    print("\n[Corpus Evaluation on Brown Corpus...]")
    brown_sents = list(brown.sents())
    stats = evaluate_on_corpus(analyzer, brown_sents, n=3000)

    print(f"  Total words analyzed: {stats['total_words']}")
    print(f"  Words with prefix:    {stats['words_with_prefix']}")
    print(f"  Words with suffix:    {stats['words_with_suffix']}")
    print(f"  POS distribution:     {stats['pos_distribution']}")
    print(f"  Top suffixes:         {list(stats['top_suffixes'].items())[:5]}")
    print(f"  Top prefixes:         {list(stats['top_prefixes'].items())[:5]}")

    plot_pos_distribution(stats['pos_distribution'])
    plot_suffix_counts(stats['top_suffixes'])
    plot_prefix_counts(stats['top_prefixes'])

    # Sample detailed analysis
    sample_words = ['unhappiness', 'internationalization', 'reprocessing']
    print("\n[Detailed Analysis Samples]")
    for word in sample_words:
        a = analyzer.analyze(word)
        print(f"\n  Word: {word}")
        print(f"    Morpheme count : {a['num_morphemes']}")
        for p in a['prefixes']:
            print(f"    Prefix: {p['prefix']:<8} → {p['description']}")
        for s in a['suffixes'][:2]:
            print(f"    Suffix: {s['suffix']:<8} → stem: {s['stem']:<12} POS: {s['pos']}")

    save_metrics(stats, os.path.join(METRICS_DIR, 'cat3_rule_morphology.json'))
    print("\n  Metrics saved to outputs/metrics/cat3_rule_morphology.json")
    print("\n[CATEGORY 3 COMPLETE]")
    return stats


if __name__ == '__main__':
    run()
