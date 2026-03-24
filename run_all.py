#!/usr/bin/env python3
"""
run_all.py — Master execution script for all NLP Lab categories.
Runs experiments 1–9 in sequence, reporting status and timing.

Usage:
    python run_all.py              # Run all categories
    python run_all.py --cat 1 5 7  # Run specific categories
    python run_all.py --skip 7     # Skip specific categories
"""

import sys
import os
import time
import argparse
import traceback
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CATEGORIES = {
    1: ('cat1_preprocessing',      'Text Preprocessing'),
    2: ('cat2_entropy_perplexity', 'Entropy, Cross-Entropy, Perplexity'),
    3: ('cat3_rule_morphology',    'Rule-Based Morphological Analyzer'),
    4: ('cat4_fsa_morphology',     'FSA-Based Morphological Analyzer'),
    5: ('cat5_unigram_model',      'Unigram Language Model'),
    6: ('cat6_bigram_model',       'Bigram Language Model'),
    7: ('cat7_neural_models',      'Neural Language Models'),
    8: ('cat8_vector_semantics',   'Vector Semantics'),
    9: ('cat9_word_embeddings',    'Word Embeddings + Visualization'),
}


def run_category(cat_num: int) -> tuple:
    """Run a single category. Returns (success, elapsed_seconds, error_msg)."""
    folder, name = CATEGORIES[cat_num]
    module_path = os.path.join(ROOT, 'experiments', folder, 'run.py')

    if not os.path.exists(module_path):
        return False, 0.0, f"run.py not found at {module_path}"

    start = time.time()
    try:
        import importlib.util
        spec   = importlib.util.spec_from_file_location(f"cat{cat_num}_run", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run()
        elapsed = time.time() - start
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        return False, elapsed, f"{type(e).__name__}: {e}\n{tb}"


def print_banner():
    print("\n" + "=" * 70)
    print("  NLP LAB REPOSITORY — FULL EXPERIMENT SUITE")
    print("  Covers Categories 1–9: Preprocessing → Embeddings")
    print("=" * 70)


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  {'Cat':<5} {'Name':<40} {'Status':<10} {'Time':>8}")
    print(f"  {'-'*65}")
    total_time = 0.0
    passed = 0
    for cat_num, (folder, name) in CATEGORIES.items():
        if cat_num not in results:
            continue
        success, elapsed, error = results[cat_num]
        status = 'PASS' if success else 'FAIL'
        total_time += elapsed
        if success:
            passed += 1
        print(f"  {cat_num:<5} {name:<40} {status:<10} {elapsed:>6.1f}s")

    print(f"\n  Total: {passed}/{len(results)} passed  |  "
          f"Elapsed: {total_time:.1f}s")

    # Print any errors
    failures = [(n, r) for n, r in results.items() if not r[0]]
    if failures:
        print("\n  ERRORS:")
        for cat_num, (_, _, error) in failures:
            print(f"\n  --- Category {cat_num} ---")
            # Print only first 10 lines of traceback
            lines = (error or '').splitlines()[:10]
            for line in lines:
                print(f"  {line}")

    print("=" * 70)
    return passed == len(results)


def main():
    parser = argparse.ArgumentParser(description='Run NLP Lab experiments')
    parser.add_argument('--cat', nargs='+', type=int, help='Run only these categories')
    parser.add_argument('--skip', nargs='+', type=int, help='Skip these categories')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop execution on first failure')
    args = parser.parse_args()

    print_banner()

    # Determine which categories to run
    to_run = sorted(CATEGORIES.keys())
    if args.cat:
        to_run = sorted(c for c in args.cat if c in CATEGORIES)
    if args.skip:
        to_run = [c for c in to_run if c not in args.skip]

    print(f"\n  Running categories: {to_run}")
    print(f"  Output directory: outputs/\n")

    # Install NLTK data upfront
    print("[Pre-flight: downloading NLTK corpora...]")
    try:
        import nltk
        for resource in ['brown', 'reuters', 'punkt', 'punkt_tab', 'stopwords',
                         'wordnet', 'averaged_perceptron_tagger',
                         'averaged_perceptron_tagger_eng', 'omw-1.4', 'words']:
            nltk.download(resource, quiet=True)
        print("  NLTK data ready.\n")
    except Exception as e:
        print(f"  Warning: NLTK download issue: {e}\n")

    results = {}
    for cat_num in to_run:
        folder, name = CATEGORIES[cat_num]
        print(f"\n{'─'*70}")
        print(f"  Running Category {cat_num}: {name}")
        print(f"{'─'*70}")

        success, elapsed, error = run_category(cat_num)
        results[cat_num] = (success, elapsed, error)

        if not success:
            print(f"\n  [ERROR in Category {cat_num}]")
            if error:
                for line in (error or '').splitlines()[:5]:
                    print(f"  {line}")
            if args.stop_on_error:
                print("\n  Stopping due to --stop-on-error flag.")
                break

    all_passed = print_summary(results)

    # Save run summary
    summary = {
        str(cat): {
            'name': CATEGORIES[cat][1],
            'success': results[cat][0],
            'elapsed_seconds': round(results[cat][1], 2),
        }
        for cat in results
    }
    summary_path = os.path.join(ROOT, 'outputs', 'metrics', 'run_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Run summary saved to: outputs/metrics/run_summary.json")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
