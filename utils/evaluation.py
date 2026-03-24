"""
Evaluation utilities: metrics, report generation, comparison tables.
"""

import math
import json
import csv
import os
from typing import List, Dict, Any, Tuple
import collections


# ─── Metric Functions ─────────────────────────────────────────────────────────

def accuracy(y_true: List, y_pred: List) -> float:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / max(len(y_true), 1)


def precision_recall_f1(y_true: List[str], y_pred: List[str],
                        positive_class: str = '1') -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / max(norm_a * norm_b, 1e-10)


def perplexity_from_log_probs(log_probs: List[float]) -> float:
    avg_log_prob = sum(log_probs) / max(len(log_probs), 1)
    return math.exp(-avg_log_prob)


# ─── Text Classification Metrics ──────────────────────────────────────────────

def confusion_matrix(y_true: List, y_pred: List) -> Dict:
    classes = sorted(set(y_true) | set(y_pred))
    matrix = {c: {cc: 0 for cc in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_metrics(metrics: Dict[str, Any], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(rows: List[Dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        return
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ─── Comparison Printing ──────────────────────────────────────────────────────

def print_comparison_table(data: Dict[str, Dict[str, float]], title: str = ''):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    models = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    col_w = max(max(len(m) for m in metrics), 12)
    mod_w = max(max(len(m) for m in models), 15)
    header = f"{'Model':<{mod_w}}" + ''.join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print('-' * len(header))
    for model, vals in data.items():
        row = f"{model:<{mod_w}}" + ''.join(f"{vals.get(m, 0):>{col_w}.4f}" for m in metrics)
        print(row)
    print()


def word_freq_table(counts: Dict[str, int], top_n: int = 20) -> List[Dict]:
    total = sum(counts.values())
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    return [{'word': w, 'count': c, 'probability': c / total}
            for w, c in sorted_items]
