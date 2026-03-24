"""
Category 4: FSA-Based Morphological Analyzer
=============================================
Aim: Model morphological word structure using Finite State Automata (FSA).
Objective: Implement FSA state transitions for English morphology —
           pluralization, past tense, progressive, and comparatives.
Dataset: Brown Corpus word samples
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import json
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import nltk
from nltk.corpus import brown

from utils.evaluation import save_metrics

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

# ─── FSA State Machine ────────────────────────────────────────────────────────

class FSAState:
    """A state in the FSA."""
    def __init__(self, name: str, is_accepting: bool = False, label: str = ''):
        self.name = name
        self.is_accepting = is_accepting
        self.label = label
        self.transitions: dict = {}  # char/condition → next_state

    def add_transition(self, symbol: str, next_state: 'FSAState'):
        self.transitions[symbol] = next_state

    def __repr__(self):
        accept = '*' if self.is_accepting else ''
        return f"State({self.name}{accept})"


class FSA:
    """
    Finite State Automaton for morphological analysis.
    Supports:
      - run(input_string) → bool (accepts/rejects)
      - get_path(input_string) → list of states traversed
    """

    def __init__(self, name: str):
        self.name = name
        self.states: dict = {}
        self.start_state: FSAState = None
        self.accept_states: set = set()

    def add_state(self, state: FSAState):
        self.states[state.name] = state
        if state.is_accepting:
            self.accept_states.add(state.name)

    def set_start(self, name: str):
        self.start_state = self.states[name]

    def run(self, word: str) -> bool:
        """Deterministic FSA run — returns True if word is accepted."""
        current = self.start_state
        if current is None:
            return False
        for char in word:
            if char in current.transitions:
                current = current.transitions[char]
            elif '*' in current.transitions:
                current = current.transitions['*']
            else:
                return False
        return current.name in self.accept_states

    def get_path(self, word: str) -> list:
        """Return list of (state, char) pairs for visualization."""
        current = self.start_state
        path = [current.name]
        for char in word:
            if char in current.transitions:
                current = current.transitions[char]
                path.append(current.name)
            elif '*' in current.transitions:
                current = current.transitions['*']
                path.append(current.name)
            else:
                path.append('REJECT')
                break
        return path


# ─── Morphological FSA Builders ──────────────────────────────────────────────

class MorphologicalFSA:
    """
    Collection of FSAs for English morphological patterns.
    Uses regex-based FSA simulation for practical efficiency.
    """

    # Pluralization patterns — use fullmatch patterns (anchored start+end)
    PLURAL_RULES = [
        # (pattern, handler, example)
        (r'.*(?:s|sh|ch|x|z)$',  lambda w: w + 'es',           'bus→buses'),
        (r'.*[^aeiou]y$',         lambda w: w[:-1] + 'ies',     'city→cities'),
        (r'.*[aeiou]y$',          lambda w: w + 's',            'day→days'),
        (r'.*(?:leaf|loaf|half)$',lambda w: w[:-1] + 'ves',     'leaf→leaves'),
        (r'.*fe$',                lambda w: w[:-2] + 'ves',     'knife→knives'),
        (r'.*',                   lambda w: w + 's',            'cat→cats'),
    ]

    # Past tense patterns
    PAST_TENSE_RULES = [
        (r'.*e$',           lambda w: w + 'd',            'bake→baked'),
        (r'.*[^aeiou]y$',   lambda w: w[:-1] + 'ied',     'carry→carried'),
        (r'.*[aeiou][bcdfgklmnprst]$',
                            lambda w: w + w[-1] + 'ed',   'run→runned'),
        (r'.*',             lambda w: w + 'ed',           'walk→walked'),
    ]

    # Progressive (ing) patterns
    PROGRESSIVE_RULES = [
        (r'.*[^aeiou]e$',  lambda w: w[:-1] + 'ing',       'make→making'),
        (r'.*[aeiou][bcdfgklmnprst]$',
                           lambda w: w + w[-1] + 'ing',    'run→running'),
        (r'.*',            lambda w: w + 'ing',             'walk→walking'),
    ]

    def __init__(self):
        self._build_fsas()

    def _build_fsas(self):
        """Build FSAs for each morphological process."""
        # We build a character-level FSA for suffix detection
        self.plural_fsa       = self._build_suffix_fsa(['s', 'es', 'ies', 'ves'])
        self.past_fsa         = self._build_suffix_fsa(['ed', 'ied', 'd'])
        self.progressive_fsa  = self._build_suffix_fsa(['ing'])
        self.comparative_fsa  = self._build_suffix_fsa(['er', 'est'])
        self.negation_fsa     = self._build_prefix_fsa(['un', 'dis', 'in', 'non', 'im'])

    def _build_suffix_fsa(self, suffixes: list) -> FSA:
        """Build a simple FSA that recognizes words ending in any of the given suffixes."""
        fsa = FSA(name=f"suffix_{'-'.join(suffixes)}")

        # Create start state
        start = FSAState('START')
        fsa.add_state(start)
        fsa.set_start('START')

        # "Anything" state — consumes arbitrary characters
        mid = FSAState('MID')
        mid.add_transition('*', mid)  # self-loop on any char
        fsa.add_state(mid)
        start.add_transition('*', mid)

        # Build suffix DFA for each suffix
        for suffix in suffixes:
            prev = mid
            states_chain = []
            for i, char in enumerate(suffix):
                state_name = f"S_{suffix}_{i}"
                is_final = (i == len(suffix) - 1)
                state = FSAState(state_name, is_accepting=is_final,
                                 label=f"...{suffix[:i+1]}")
                fsa.add_state(state)
                prev.add_transition(char, state)
                states_chain.append(state)
                prev = state
            # Allow the last suffix state to also loop back for longer suffixes
            if states_chain:
                states_chain[-1].add_transition('*', mid)

        return fsa

    def _build_prefix_fsa(self, prefixes: list) -> FSA:
        """Build FSA recognizing words starting with any of the given prefixes."""
        fsa = FSA(name=f"prefix_{'-'.join(prefixes)}")

        start = FSAState('START')
        fsa.add_state(start)
        fsa.set_start('START')

        for prefix in prefixes:
            prev = start
            for i, char in enumerate(prefix):
                state_name = f"P_{prefix}_{i}"
                is_final = (i == len(prefix) - 1)
                state = FSAState(state_name, is_accepting=is_final)
                fsa.add_state(state)
                prev.add_transition(char, state)
                prev = state
            # After accepting prefix, accept rest of word
            rest = FSAState(f"P_{prefix}_REST", is_accepting=True)
            rest.add_transition('*', rest)
            fsa.add_state(rest)
            prev.add_transition('*', rest)

        return fsa

    # ── Apply Rules ──────────────────────────────────────────────────────────

    def pluralize(self, word: str) -> str:
        w = word.lower()
        for pattern, handler, _ in self.PLURAL_RULES:
            if re.fullmatch(pattern, w):
                return handler(w)
        return w + 's'

    def past_tense(self, word: str) -> str:
        w = word.lower()
        for pattern, handler, _ in self.PAST_TENSE_RULES:
            if re.fullmatch(pattern, w):
                return handler(w)
        return w + 'ed'

    def progressive(self, word: str) -> str:
        w = word.lower()
        for pattern, handler, _ in self.PROGRESSIVE_RULES:
            if re.fullmatch(pattern, w):
                return handler(w)
        return w + 'ing'

    def detect_form(self, word: str) -> dict:
        """Detect morphological form of a word using FSAs."""
        w = word.lower()
        return {
            'word': word,
            'plural':      self.plural_fsa.run(w),
            'past_tense':  self.past_fsa.run(w),
            'progressive': self.progressive_fsa.run(w),
            'comparative': self.comparative_fsa.run(w),
            'negation':    self.negation_fsa.run(w),
        }

# ─── Visualization ────────────────────────────────────────────────────────────

def draw_fsa_diagram(states: list, transitions: list, accepting: list, title: str, filename: str):
    """Draw a simple FSA diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, len(states) - 0.5)
    ax.set_ylim(-1, 2)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold')

    pos = {s: (i, 0.5) for i, s in enumerate(states)}
    radius = 0.25

    # Draw states
    for state in states:
        x, y = pos[state]
        color = '#90EE90' if state in accepting else '#AED6F1'
        circle = plt.Circle((x, y), radius, color=color, ec='black', zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, state, ha='center', va='center', fontsize=8, zorder=4)
        if state == states[0]:
            ax.annotate('', xy=(x - radius, y), xytext=(x - 0.5, y),
                        arrowprops=dict(arrowstyle='->', color='black'))

    # Draw transitions
    for (src, sym, dst) in transitions:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        if src == dst:
            ax.annotate(sym, xy=(x1, y1 + radius + 0.05), fontsize=7,
                        ha='center', color='darkred')
        else:
            mid_x = (x1 + x2) / 2
            ax.annotate('', xy=(x2 - radius, y2), xytext=(x1 + radius, y1),
                        arrowprops=dict(arrowstyle='->', color='navy', lw=1.2))
            ax.text(mid_x, y1 + 0.15, sym, ha='center', fontsize=7, color='darkblue')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_morpheme_detection(detection_results: dict):
    """Bar chart of how many words match each morphological category."""
    categories = list(detection_results.keys())
    counts = list(detection_results.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(categories, counts, color=sns.color_palette('Set3', len(categories)))
    ax.set_title('FSA Morphological Category Detection (Brown Corpus)')
    ax.set_ylabel('Number of Words Detected')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat4_morpheme_detection.png'), dpi=150)
    plt.close()
    print("  Saved: cat4_morpheme_detection.png")


import seaborn as sns

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 4: FSA-BASED MORPHOLOGICAL ANALYZER")
    print("=" * 60)

    nltk.download('brown', quiet=True)

    morph = MorphologicalFSA()

    # Demo: Generation
    print("\n[Demo: Morphological Generation]")
    demo_words = ['walk', 'run', 'carry', 'bake', 'city', 'leaf', 'bus', 'knife']
    print(f"  {'Word':<12} {'Plural':<15} {'Past Tense':<15} {'Progressive':<15}")
    print(f"  {'-'*57}")
    for word in demo_words:
        plural      = morph.pluralize(word)
        past        = morph.past_tense(word)
        progressive = morph.progressive(word)
        print(f"  {word:<12} {plural:<15} {past:<15} {progressive:<15}")

    # Demo: Detection via FSA
    print("\n[Demo: Morphological Detection via FSA]")
    detect_words = [
        'running', 'walked', 'cities', 'unhappy', 'quickly',
        'displacement', 'impossible', 'international',
    ]
    for word in detect_words:
        result = morph.detect_form(word)
        flags = [k for k, v in result.items() if v and k != 'word']
        print(f"  {word:<25} detected forms: {', '.join(flags) or 'none'}")

    # Draw FSA diagram for plural suffix FSA
    draw_fsa_diagram(
        states=['START', 'MID', 'ACCEPT_s', 'ACCEPT_es'],
        transitions=[
            ('START', '*', 'MID'),
            ('MID', '*', 'MID'),
            ('MID', 's', 'ACCEPT_s'),
            ('MID', 'e→s', 'ACCEPT_es'),
        ],
        accepting=['ACCEPT_s', 'ACCEPT_es'],
        title='Simplified FSA: Plural Suffix Detection (-s, -es)',
        filename='cat4_fsa_plural.png',
    )

    draw_fsa_diagram(
        states=['START', 'MID', 'ACCEPT_ing'],
        transitions=[
            ('START', '*', 'MID'),
            ('MID', '*', 'MID'),
            ('MID', 'i→n→g', 'ACCEPT_ing'),
        ],
        accepting=['ACCEPT_ing'],
        title='Simplified FSA: Progressive Suffix Detection (-ing)',
        filename='cat4_fsa_progressive.png',
    )

    # Corpus-level detection statistics
    print("\n[Corpus Evaluation on Brown Corpus...]")
    brown_words = set()
    for sent in list(brown.sents())[:3000]:
        for w in sent:
            if w.isalpha() and len(w) > 3:
                brown_words.add(w.lower())

    brown_words_list = list(brown_words)[:2000]
    detection_counts = {
        'plural': 0, 'past_tense': 0, 'progressive': 0,
        'comparative': 0, 'negation': 0,
    }
    for word in brown_words_list:
        result = morph.detect_form(word)
        for cat in detection_counts:
            if result[cat]:
                detection_counts[cat] += 1

    print(f"  Words analyzed: {len(brown_words_list)}")
    for cat, count in detection_counts.items():
        pct = count / len(brown_words_list) * 100
        print(f"  {cat:<15}: {count:>5} ({pct:.1f}%)")

    plot_morpheme_detection(detection_counts)

    # Specific examples with state paths
    print("\n[FSA State Paths for Sample Words]")
    for word in ['running', 'walked', 'cities']:
        path = morph.progressive_fsa.get_path(word) if word.endswith('ing') \
               else morph.past_fsa.get_path(word) if word.endswith('ed') \
               else morph.plural_fsa.get_path(word)
        print(f"  {word}: {' → '.join(path[:6])}{'...' if len(path)>6 else ''}")

    save_metrics({
        'words_analyzed': len(brown_words_list),
        'detection_counts': detection_counts,
        'detection_rates': {
            k: round(v / len(brown_words_list), 4)
            for k, v in detection_counts.items()
        }
    }, os.path.join(METRICS_DIR, 'cat4_fsa_morphology.json'))
    print("\n  Metrics saved to outputs/metrics/cat4_fsa_morphology.json")
    print("\n[CATEGORY 4 COMPLETE]")


if __name__ == '__main__':
    run()
