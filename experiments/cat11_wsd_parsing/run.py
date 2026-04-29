"""
Category 11: Word Sense Disambiguation + Dependency/Constituency Parsing
=========================================================================
Exercise 1 — Word Sense Disambiguation using the Lesk algorithm
Exercise 2 — Constituency Parsing (CFG/ChartParser) and
             Dependency Parsing (spaCy or rule-based fallback)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import collections
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import nltk

from utils.evaluation import save_metrics

# ─── Output directories ───────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ─── NLTK data downloads ──────────────────────────────────────────────────────
def download_data():
    for resource in [
        'wordnet', 'brown', 'averaged_perceptron_tagger',
        'punkt', 'omw-1.4', 'stopwords', 'punkt_tab',
        'averaged_perceptron_tagger_eng',
    ]:
        nltk.download(resource, quiet=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1 — Word Sense Disambiguation
# ═══════════════════════════════════════════════════════════════════════════════

# WSD_TEST_CASES third field is a human-readable hint for documentation; not used in logic.
WSD_TEST_CASES = [
    # bank – financial
    ("I deposited money in the bank yesterday.",                          "bank", "financial"),
    ("She withdrew cash from the bank account.",                         "bank", "financial"),
    # bank – river
    ("We sat on the grassy bank beside the river.",                      "bank", "slope"),
    ("The fisherman cast his line from the river bank.",                 "bank", "slope"),
    # bat – cricket / sports
    ("The baseball player swung the bat and hit a home run.",            "bat",  "cricket_bat"),
    ("He bought a new wooden bat for the game.",                         "bat",  "cricket_bat"),
    # bat – animal
    ("A bat flew out of the cave at dusk.",                              "bat",  "chiropteran"),
    ("The bat hung upside down from the ceiling of the cave.",           "bat",  "chiropteran"),
    # plant – vegetation
    ("She watered the plant on the windowsill every morning.",           "plant", "flora"),
    ("The gardener pruned each plant carefully.",                        "plant", "flora"),
    # plant – factory
    ("The manufacturing plant was shut down due to pollution.",          "plant", "industrial_plant"),
    ("Workers at the chemical plant went on strike.",                    "plant", "industrial_plant"),
    # bass – music
    ("He plays bass guitar in a jazz band.",                             "bass", "bass_guitar"),
    ("The bass notes resonated throughout the concert hall.",            "bass", "bass_voice"),
    # bass – fish
    ("They caught a large striped bass near the pier.",                  "bass", "bass_fish"),
    ("The fisherman was proud of the bass he reeled in.",                "bass", "bass_fish"),
    # crane – bird
    ("A crane waded through the shallow marsh at sunrise.",              "crane", "crane_bird"),
    ("The elegant crane stood motionless by the lake.",                  "crane", "crane_bird"),
    # crane – machine
    ("A construction crane lifted the steel beam high above the street.", "crane", "crane_machine"),
    ("The tower crane dominated the city skyline.",                      "crane", "crane_machine"),
]

# Human-readable expected sense labels per test case (same order)
WSD_EXPECTED_LABELS = [
    "financial_institution", "financial_institution",
    "river_bank",            "river_bank",
    "cricket_bat",           "cricket_bat",
    "animal_bat",            "animal_bat",
    "vegetation",            "vegetation",
    "factory",               "factory",
    "bass_music",            "bass_music",
    "bass_fish",             "bass_fish",
    "crane_bird",            "crane_bird",
    "crane_machine",         "crane_machine",
]

# WordNet synset name fragments that map to the two senses for each word.
# Keys MUST match the strings used in WSD_EXPECTED_LABELS.
SENSE_MAP = {
    "bank":  {
        "financial_institution": ["depository_financial", "savings_bank", "bank.n.02",
                                  "bank.n.04", "bank.n.05", "bank.n.06"],
        "river_bank":            ["bank.n.01", "bank.n.03", "bank.n.07", "slope"],
    },
    "bat":   {
        "cricket_bat":  ["cricket_bat", "baseball_bat", "bat.n.02",
                         "squash_racket", "racket", "club"],
        "animal_bat":   ["bat.n.01", "chiroptera", "chiropteran"],
    },
    "plant": {
        "vegetation": ["plant.n.02", "flora", "plant.n.01",
                       "organism", "life_form"],
        "factory":    ["industrial_plant", "plant.n.03", "works", "mill"],
    },
    "bass":  {
        "bass_music": ["bass.n.01", "bass.n.02", "bass_guitar",
                       "bass_voice", "bass.n.03", "bass.n.05",
                       "musical", "voice", "singer"],
        "bass_fish":  ["bass.n.04", "bass.n.06", "largemouth",
                       "smallmouth", "striped_bass", "fresh_water_fish",
                       "saltwater_fish"],
    },
    "crane": {
        "crane_bird":    ["crane.n.01", "crane.n.02", "wading_bird", "bird"],
        "crane_machine": ["crane.n.03", "derrick", "crane.v.01", "machine",
                          "lifting_device", "equipment"],
    },
}


def _sense_label(word: str, synset) -> str:
    """Return a human-readable label for a synset given the target word."""
    if synset is None:
        return "unknown"
    sname = synset.name().lower()
    sdef  = synset.definition().lower()
    combined = sname + " " + sdef
    if word not in SENSE_MAP:
        return sname
    for label, fragments in SENSE_MAP[word].items():
        for frag in fragments:
            if frag.lower() in combined:
                return label
    return sname.split(".")[0]


# ─── Simplified Lesk from scratch ────────────────────────────────────────────

def simple_lesk(sentence: str, target_word: str):
    """
    A from-scratch Simplified Lesk:
    overlap = |gloss_words ∩ context_words|, pick synset with max overlap.
    """
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords

    stop = set(stopwords.words('english'))
    ctx_tokens = set(re.findall(r'\b\w+\b', sentence.lower())) - stop - {target_word.lower()}

    best_synset  = None
    best_overlap = -1

    for synset in wn.synsets(target_word):
        gloss = synset.definition()
        for example in synset.examples():
            gloss += " " + example
        # also include hypernym glosses
        for hyper in synset.hypernyms():
            gloss += " " + hyper.definition()
        gloss_tokens = set(re.findall(r'\b\w+\b', gloss.lower())) - stop

        overlap = len(ctx_tokens & gloss_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_synset  = synset

    return best_synset


def run_wsd():
    """Run WSD experiments and return result dicts."""
    from nltk.wsd  import lesk
    from nltk.corpus import wordnet as wn

    nltk_results   = []  # (sentence, word, synset, label)
    simple_results = []

    for (sent, word, _), expected in zip(WSD_TEST_CASES, WSD_EXPECTED_LABELS):
        ctx_tokens = sent.lower().split()

        # NLTK Lesk
        nltk_syn   = lesk(ctx_tokens, word, pos='n')
        nltk_label = _sense_label(word, nltk_syn)
        nltk_results.append({
            "sentence": sent, "word": word,
            "synset":   nltk_syn.name() if nltk_syn else "None",
            "label":    nltk_label, "expected": expected
        })

        # Simple Lesk
        simp_syn   = simple_lesk(sent, word)
        simp_label = _sense_label(word, simp_syn)
        simple_results.append({
            "sentence": sent, "word": word,
            "synset":   simp_syn.name() if simp_syn else "None",
            "label":    simp_label, "expected": expected
        })

    return nltk_results, simple_results


def wsd_accuracy(results):
    """Fraction where label matches expected."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["label"] == r["expected"])
    return correct / len(results)


def plot_wsd_sense_distribution(nltk_results, simple_results):
    """Bar chart of predicted sense labels per target word (NLTK Lesk)."""
    words = ["bank", "bat", "plant", "bass", "crane"]
    sense_counts = {w: collections.Counter() for w in words}
    for r in nltk_results:
        sense_counts[r["word"]][r["label"]] += 1

    fig, axes = plt.subplots(1, len(words), figsize=(16, 4), sharey=False)
    fig.suptitle("WSD — Sense Distribution per Target Word (NLTK Lesk)", fontsize=13)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for ax, word, color in zip(axes, words, colors):
        counter = sense_counts[word]
        labels  = list(counter.keys())
        vals    = list(counter.values())
        ax.bar(labels, vals, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(word.upper(), fontsize=11)
        ax.set_ylabel("Count")
        ax.set_xlabel("Predicted sense")
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cat11_wsd_sense_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_wsd_accuracy_comparison(nltk_acc, simple_acc):
    """Bar chart comparing WSD method accuracies."""
    methods = ["NLTK Lesk", "Simple Lesk\n(from scratch)"]
    accs    = [nltk_acc, simple_acc]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, [a * 100 for a in accs],
                  color=["#4C72B0", "#DD8452"], edgecolor='black', alpha=0.85, width=0.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("WSD Method Accuracy Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc*100:.1f}%", ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=50, color='grey', linestyle='--', alpha=0.5, label="Baseline (random)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cat11_wsd_lesk_accuracy.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 — Parsing
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Constituency Parsing ────────────────────────────────────────────────────

GRAMMAR_STR = """
    S   -> NP VP
    S   -> NP VP PP
    NP  -> DT NN
    NP  -> DT JJ NN
    NP  -> PRP
    NP  -> NNP
    NP  -> NNP NNP
    NP  -> DT NN NN
    VP  -> VBZ NP
    VP  -> VBD NP
    VP  -> VBZ PP
    VP  -> VBD PP
    VP  -> VBP NP
    VP  -> MD VB NP
    VP  -> VBZ NP PP
    VP  -> VBD NP PP
    VP  -> VBP NP PP
    PP  -> IN NP
    DT  -> 'the' | 'a' | 'an' | 'The' | 'A'
    NN  -> 'dog' | 'cat' | 'man' | 'woman' | 'bank' | 'river' | 'book' | 'car' | 'tree' | 'bird' | 'bat' | 'crane' | 'plant'
    NNP -> 'John' | 'Mary' | 'London' | 'Alice' | 'Bob'
    JJ  -> 'big' | 'small' | 'old' | 'young' | 'tall' | 'little' | 'fascinating' | 'wooden'
    PRP -> 'He' | 'She' | 'It' | 'They' | 'he' | 'she' | 'it'
    VBZ -> 'sees' | 'likes' | 'loves' | 'reads' | 'eats'
    VBD -> 'saw' | 'liked' | 'loved' | 'read' | 'ate' | 'chased'
    VBP -> 'see' | 'like' | 'love' | 'read' | 'eat'
    VB  -> 'see' | 'like' | 'love' | 'read' | 'eat'
    MD  -> 'can' | 'will' | 'may' | 'should'
    IN  -> 'in' | 'on' | 'at' | 'near' | 'by' | 'with'
"""

CONSTITUENCY_SENTENCES = [
    "the dog sees a cat",
    "John loves Mary",
    "She reads a big book",
    "the man saw a bird",
    "Alice chased the old cat",
    "He eats a small plant",
    "the crane eats a bat",
    "Mary likes the tall tree",
]


def run_constituency_parsing():
    """Parse sentences with NLTK ChartParser; return list of (sent, tree_str)."""
    grammar = nltk.CFG.fromstring(GRAMMAR_STR)
    parser  = nltk.ChartParser(grammar)

    results = []
    for sent in CONSTITUENCY_SENTENCES:
        tokens = sent.split()
        trees  = list(parser.parse(tokens))
        if trees:
            tree_str = trees[0].pformat(margin=60)
            results.append((sent, tree_str, trees[0]))
        else:
            results.append((sent, "(no parse)", None))
    return results


def _draw_tree_text(tree, prefix="", is_last=True, lines=None):
    """Render an NLTK Tree as ASCII art into a list of strings."""
    if lines is None:
        lines = []
    connector  = "└── " if is_last else "├── "
    child_pfx  = prefix + ("    " if is_last else "│   ")

    label = tree.label() if hasattr(tree, 'label') else str(tree)
    lines.append(prefix + connector + label)

    if hasattr(tree, '__iter__') and not isinstance(tree, str):
        children = list(tree)
        for i, child in enumerate(children):
            last = (i == len(children) - 1)
            if isinstance(child, str):
                conn2 = "└── " if last else "├── "
                lines.append(child_pfx + conn2 + f'"{child}"')
            else:
                _draw_tree_text(child, child_pfx, last, lines)
    return lines


def plot_constituency_tree(parse_results):
    """Save a text-art rendering of parse trees as an image."""
    fig, axes = plt.subplots(len(parse_results), 1,
                             figsize=(10, max(3 * len(parse_results), 8)))
    if len(parse_results) == 1:
        axes = [axes]

    fig.suptitle("Constituency Parse Trees (CFG + ChartParser)", fontsize=13, y=1.0)

    for ax, (sent, tree_str, tree_obj) in zip(axes, parse_results):
        ax.axis('off')
        if tree_obj is not None:
            lines = ["S"]
            _draw_tree_text(tree_obj, "", True, lines)
            text = "\n".join(lines)
        else:
            text = tree_str
        ax.text(0.02, 0.95, f'"{sent}"\n\n{text}',
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cat11_constituency_tree.png")
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Dependency Parsing ──────────────────────────────────────────────────────

DEP_SENTENCES = [
    "The dog chased the cat in the garden.",
    "Alice reads a fascinating book about history.",
    "John can see the tall crane near the river.",
    "The old man sat on a wooden bench.",
    "She loves the little bird by the tree.",
]


def _spacy_dep_parse(sentences):
    """Use spaCy for dependency parsing; returns list of dep triple lists."""
    import spacy
    nlp = spacy.load('en_core_web_sm')
    results = []
    for sent in sentences:
        doc = nlp(sent)
        triples = [(token.head.text, token.dep_, token.text)
                   for token in doc if token.dep_ != 'ROOT']
        root = next((t.text for t in doc if t.dep_ == 'ROOT'), "?")
        results.append({"sentence": sent, "root": root, "triples": triples})
    return results


def _rule_dep_parse(sentences):
    """
    Simple rule-based dependency parser using NLTK POS tags.
    Implements a very light left-arc / right-arc heuristic:
    - The main verb is the ROOT.
    - The leftmost NP before the verb is nsubj.
    - The first NP after the verb is dobj.
    - DT before a noun is det.
    - JJ before a noun is amod.
    - IN introduces a PP; noun after IN is pobj, IN depends on preceding noun/verb.
    """
    results = []
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)

        triples = []
        root_idx = None

        # Find main verb (VBD, VBZ, VBP, MD)
        for i, (word, tag) in enumerate(tagged):
            if tag in ('VBD', 'VBZ', 'VBP', 'MD', 'VB'):
                root_idx = i
                break
        if root_idx is None:
            root_idx = 0

        root_word = tagged[root_idx][0]

        # nsubj: first NN/NNP/PRP to the left of root
        for i in range(root_idx - 1, -1, -1):
            w, t = tagged[i]
            if t in ('NN', 'NNS', 'NNP', 'NNPS', 'PRP'):
                triples.append((root_word, "nsubj", w))
                break

        # dobj: first NN/NNP/PRP to the right of root (before any IN)
        for i in range(root_idx + 1, len(tagged)):
            w, t = tagged[i]
            if t == 'IN':
                break
            if t in ('NN', 'NNS', 'NNP', 'NNPS', 'PRP'):
                triples.append((root_word, "dobj", w))
                break

        # det / amod: scan all nouns
        for i, (word, tag) in enumerate(tagged):
            if tag in ('NN', 'NNS', 'NNP', 'NNPS'):
                if i > 0 and tagged[i-1][1] == 'DT':
                    triples.append((word, "det", tagged[i-1][0]))
                if i > 0 and tagged[i-1][1] in ('JJ', 'JJS', 'JJR'):
                    triples.append((word, "amod", tagged[i-1][0]))
                if i > 1 and tagged[i-2][1] == 'DT' and tagged[i-1][1] in ('JJ',):
                    triples.append((word, "det", tagged[i-2][0]))

        # prep/pobj: find IN tokens
        for i, (word, tag) in enumerate(tagged):
            if tag == 'IN':
                # head of prep: preceding noun or verb
                head = None
                for j in range(i - 1, -1, -1):
                    w2, t2 = tagged[j]
                    if t2 in ('NN', 'NNS', 'NNP', 'VBD', 'VBZ', 'VBP', 'VB'):
                        head = w2
                        break
                if head:
                    triples.append((head, "prep", word))
                # pobj: first noun after IN
                for j in range(i + 1, len(tagged)):
                    w2, t2 = tagged[j]
                    if t2 in ('NN', 'NNS', 'NNP', 'NNPS'):
                        triples.append((word, "pobj", w2))
                        break

        results.append({"sentence": sent, "root": root_word, "triples": triples})
    return results


def run_dependency_parsing():
    """Try spaCy; fall back to rule-based parser."""
    try:
        results = _spacy_dep_parse(DEP_SENTENCES)
        method  = "spaCy (en_core_web_sm)"
    except Exception:
        results = _rule_dep_parse(DEP_SENTENCES)
        method  = "Rule-based (NLTK POS + heuristics)"
    return results, method


_ARC_HEIGHT_SCALE = 0.12   # controls how tall arcs grow with token distance
_ARC_BASE_HEIGHT  = 0.20   # minimum arc height


def plot_dependency_arcs(dep_results, method):
    """
    Arc diagram for the first sentence, then a summary bar chart
    of the most common dependency relations across all sentences.
    """
    # ── Panel 1: arc diagram for sentence 1 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Dependency Parsing  [{method}]", fontsize=12)

    ax1 = axes[0]
    first = dep_results[0]
    sent_text = first["sentence"]
    tokens    = nltk.word_tokenize(sent_text)
    triples   = first["triples"]

    ax1.set_title(f'Arc diagram — "{sent_text}"', fontsize=9)
    ax1.set_xlim(-0.5, len(tokens) - 0.5)
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=30, ha='right', fontsize=9)
    ax1.set_yticks([])

    idx_map = {t: i for i, t in enumerate(tokens)}

    drawn = set()
    for head, rel, dep in triples:
        hi = idx_map.get(head)
        di = idx_map.get(dep)
        if hi is None or di is None:
            continue
        key = (hi, di)
        if key in drawn:
            continue
        drawn.add(key)
        mid    = (hi + di) / 2
        height = abs(di - hi) * _ARC_HEIGHT_SCALE + _ARC_BASE_HEIGHT
        arc_x = np.linspace(hi, di, 60)
        arc_y = height * np.sin(np.pi * (arc_x - hi) / (di - hi + 1e-9))
        ax1.plot(arc_x, arc_y, linewidth=1.5, alpha=0.75)
        ax1.annotate("", xy=(di, 0.02),
                     xytext=(di, arc_y[30] if len(arc_y) > 30 else 0.1),
                     arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))
        ax1.text(mid, height + 0.02, rel,
                 ha='center', va='bottom', fontsize=7, color='#333')

    # mark root
    root = first["root"]
    if root in idx_map:
        ri = idx_map[root]
        ax1.text(ri, -0.08, "ROOT", ha='center', fontsize=8,
                 color='crimson', fontweight='bold')

    # ── Panel 2: bar chart of relation frequencies ────────────────────────────
    ax2 = axes[1]
    all_rels = collections.Counter()
    for r in dep_results:
        for head, rel, dep in r["triples"]:
            all_rels[rel] += 1

    labels = list(all_rels.keys())
    counts = list(all_rels.values())
    sorted_pairs = sorted(zip(counts, labels), reverse=True)
    counts, labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    colors_bar = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    ax2.barh(labels, counts, color=colors_bar, edgecolor='black', alpha=0.85)
    ax2.set_xlabel("Frequency")
    ax2.set_title("Dependency Relation Frequencies (all sentences)")
    ax2.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cat11_dependency_arcs.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_pos_vs_dep_relations(dep_results):
    """
    Stacked bar chart: for each dependency relation type,
    how many dependents fall into each broad POS category.
    """
    pos_map = {
        'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
        'VB': 'Verb', 'VBD': 'Verb', 'VBZ': 'Verb', 'VBP': 'Verb',
        'VBN': 'Verb', 'VBG': 'Verb', 'MD': 'Verb',
        'JJ': 'Adj',  'JJS': 'Adj',  'JJR': 'Adj',
        'RB': 'Adv',  'RBR': 'Adv',  'RBS': 'Adv',
        'DT': 'Det',  'IN': 'Prep',  'PRP': 'Pron', 'PRP$': 'Pron',
        'CC': 'Conj', 'CD': 'Num',
    }
    pos_cats = ['Noun', 'Verb', 'Adj', 'Adv', 'Det', 'Prep', 'Pron', 'Other']

    # Gather all dependency triples with POS of dependent
    rel_pos_counts = collections.defaultdict(lambda: collections.Counter())

    for r in dep_results:
        tokens  = nltk.word_tokenize(r["sentence"])
        tagged  = dict(nltk.pos_tag(tokens))
        for head, rel, dep in r["triples"]:
            raw_pos = tagged.get(dep, 'XX')
            cat     = pos_map.get(raw_pos, 'Other')
            rel_pos_counts[rel][cat] += 1

    rels = sorted(rel_pos_counts.keys())
    if not rels:
        return

    data = np.array([[rel_pos_counts[r][p] for p in pos_cats] for r in rels])

    fig, ax = plt.subplots(figsize=(10, max(4, len(rels) * 0.6 + 2)))
    bottom = np.zeros(len(rels))
    palette = plt.cm.Set2(np.linspace(0, 1, len(pos_cats)))

    for j, (cat, color) in enumerate(zip(pos_cats, palette)):
        vals = data[:, j]
        ax.barh(rels, vals, left=bottom, label=cat, color=color, edgecolor='white')
        bottom += vals

    ax.set_xlabel("Count")
    ax.set_title("POS Category of Dependent per Dependency Relation")
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cat11_pos_vs_dep_relations.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("Category 11: WSD + Dependency/Constituency Parsing")
    print("=" * 60)

    download_data()

    # ── Exercise 1: WSD ───────────────────────────────────────────────────────
    print("\n[Exercise 1] Word Sense Disambiguation")
    nltk_results, simple_results = run_wsd()

    nltk_acc   = wsd_accuracy(nltk_results)
    simple_acc = wsd_accuracy(simple_results)

    print(f"  NLTK Lesk  accuracy : {nltk_acc*100:.1f}%")
    print(f"  Simple Lesk accuracy: {simple_acc*100:.1f}%")

    print("\n  Sample predictions (NLTK Lesk):")
    for r in nltk_results[:6]:
        mark = "✓" if r["label"] == r["expected"] else "✗"
        print(f"  {mark} [{r['word']:5s}]  pred={r['label']:25s}  "
              f"synset={r['synset']}")

    plot_wsd_sense_distribution(nltk_results, simple_results)
    plot_wsd_accuracy_comparison(nltk_acc, simple_acc)

    # ── Exercise 2: Parsing ───────────────────────────────────────────────────
    print("\n[Exercise 2a] Constituency Parsing (CFG + ChartParser)")
    const_results = run_constituency_parsing()
    n_parsed = sum(1 for _, _, t in const_results if t is not None)
    print(f"  Parsed {n_parsed}/{len(const_results)} sentences successfully")
    for sent, tree_str, _ in const_results[:3]:
        print(f"\n  Sentence : {sent}")
        print(f"  Tree     :\n{tree_str}")

    plot_constituency_tree(const_results)

    print("\n[Exercise 2b] Dependency Parsing")
    dep_results, dep_method = run_dependency_parsing()
    print(f"  Method: {dep_method}")
    for r in dep_results[:3]:
        print(f"\n  Sentence : {r['sentence']}")
        print(f"  ROOT     : {r['root']}")
        print(f"  Triples  : {r['triples'][:5]}")

    plot_dependency_arcs(dep_results, dep_method)
    plot_pos_vs_dep_relations(dep_results)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "wsd_nltk_lesk_accuracy":   round(nltk_acc,   4),
        "wsd_simple_lesk_accuracy": round(simple_acc, 4),
        "wsd_test_cases":           len(WSD_TEST_CASES),
        "constituency_sentences":   len(CONSTITUENCY_SENTENCES),
        "constituency_parsed_ok":   n_parsed,
        "dependency_sentences":     len(DEP_SENTENCES),
        "dependency_method":        dep_method,
        "num_sentences_parsed":     n_parsed + len(DEP_SENTENCES),
        "constituency_examples":    [s for s, _, _ in const_results if _ is not None],
        "dependency_examples":      [r["sentence"] for r in dep_results],
    }
    metrics_path = os.path.join(METRICS_DIR, "cat11_wsd_parsing.json")
    save_metrics(metrics, metrics_path)
    print(f"\n  Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("Category 11 complete.")
    print("=" * 60)


if __name__ == "__main__":
    run()
