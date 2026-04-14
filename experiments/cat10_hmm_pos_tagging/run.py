"""
Category 10: Hidden Markov Model for POS Tagging
=================================================
Aim: Implement a full HMM from scratch (numpy only) for Part-of-Speech
     tagging, demonstrating Forward, Backward, Viterbi, and Baum-Welch.
Dataset: Brown Corpus (first 3000 tagged sentences)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import collections
import math
import numpy as np
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
os.makedirs(METRICS_DIR, exist_ok=True)

N_SENTENCES = 3000   # corpus subset for speed
LAPLACE_K   = 1      # Laplace smoothing constant
LOG_ZERO    = -1e18  # stand-in for log(0)


# ─── Data Loading & Preprocessing ────────────────────────────────────────────

def download_data():
    for res in ['brown', 'universal_tagset']:
        nltk.download(res, quiet=True)


def load_corpus(n: int = N_SENTENCES):
    """Return list of [(word, tag), ...] sentence lists (universal tagset)."""
    sents = list(brown.tagged_sents(tagset='universal'))[:n]
    # Lowercase words, keep tags as-is
    cleaned = [[(w.lower(), t) for w, t in s if t != '-NONE-'] for s in sents]
    cleaned = [s for s in cleaned if len(s) >= 2]
    return cleaned


def train_test_split(sents, train_frac=0.8):
    split = int(len(sents) * train_frac)
    return sents[:split], sents[split:]


# ─── HMM Parameter Estimation ────────────────────────────────────────────────

def build_hmm(train_sents):
    """
    Estimate HMM parameters from labelled sentences with Laplace smoothing.

    Returns
    -------
    tags     : list of tag strings (sorted)
    vocab    : list of word strings (sorted)
    tag2i    : dict tag -> int
    word2i   : dict word -> int
    log_pi   : (N,)    log initial probs
    log_A    : (N, N)  log transition probs
    log_B    : (N, V)  log emission probs
    """
    # Collect vocabulary and tags
    tag_counts   = collections.Counter()
    word_counts  = collections.Counter()
    for sent in train_sents:
        for w, t in sent:
            tag_counts[t]  += 1
            word_counts[w] += 1

    tags  = sorted(tag_counts.keys())
    vocab = sorted(word_counts.keys())
    N = len(tags)
    V = len(vocab)
    tag2i  = {t: i for i, t in enumerate(tags)}
    word2i = {w: i for i, w in enumerate(vocab)}

    # Count transitions and emissions
    pi_counts = np.zeros(N, dtype=float)
    A_counts  = np.zeros((N, N), dtype=float)
    B_counts  = np.zeros((N, V), dtype=float)

    for sent in train_sents:
        # Initial state
        pi_counts[tag2i[sent[0][1]]] += 1
        for k, (w, t) in enumerate(sent):
            ti = tag2i[t]
            wi = word2i.get(w, -1)
            if wi >= 0:
                B_counts[ti, wi] += 1
            if k < len(sent) - 1:
                tj = tag2i[sent[k + 1][1]]
                A_counts[ti, tj] += 1

    # Laplace smoothing and normalise to log-probabilities
    log_pi = np.log((pi_counts + LAPLACE_K) / (pi_counts.sum() + LAPLACE_K * N))
    # Transition: each row sums to 1
    A_smooth = A_counts + LAPLACE_K
    log_A = np.log(A_smooth / A_smooth.sum(axis=1, keepdims=True))
    # Emission: each row sums to 1
    B_smooth = B_counts + LAPLACE_K
    log_B = np.log(B_smooth / B_smooth.sum(axis=1, keepdims=True))

    return tags, vocab, tag2i, word2i, log_pi, log_A, log_B


# ─── Forward Algorithm ───────────────────────────────────────────────────────

def forward_log_prob(obs, tag2i, word2i, log_pi, log_A, log_B):
    """
    Compute log P(obs | model) using the Forward algorithm in log-space.

    Parameters
    ----------
    obs : list of word strings

    Returns
    -------
    log_prob : float
    alpha    : (T, N) array of log forward probabilities
    """
    N = log_pi.shape[0]
    T = len(obs)
    alpha = np.full((T, N), LOG_ZERO, dtype=float)

    # Initialise
    w0 = word2i.get(obs[0], -1)
    b0 = log_B[:, w0] if w0 >= 0 else np.zeros(N)   # uniform if OOV
    alpha[0] = log_pi + b0

    # Recursion
    for t in range(1, T):
        wt = word2i.get(obs[t], -1)
        bt = log_B[:, wt] if wt >= 0 else np.zeros(N)
        for j in range(N):
            # log-sum-exp over all i
            vals = alpha[t - 1] + log_A[:, j]
            max_v = vals.max()
            alpha[t, j] = bt[j] + max_v + np.log(np.exp(vals - max_v).sum())

    # Log-sum-exp over final states
    max_v = alpha[T - 1].max()
    log_prob = max_v + np.log(np.exp(alpha[T - 1] - max_v).sum())
    return log_prob, alpha


# ─── Backward Algorithm ──────────────────────────────────────────────────────

def backward_log(obs, word2i, log_A, log_B):
    """
    Compute log backward probabilities beta.

    Returns
    -------
    beta : (T, N) array of log backward probabilities
    """
    N = log_A.shape[0]
    T = len(obs)
    beta = np.full((T, N), LOG_ZERO, dtype=float)
    beta[T - 1] = 0.0  # log(1)

    for t in range(T - 2, -1, -1):
        wt1 = word2i.get(obs[t + 1], -1)
        bt1 = log_B[:, wt1] if wt1 >= 0 else np.zeros(N)
        for i in range(N):
            vals = log_A[i, :] + bt1 + beta[t + 1]
            max_v = vals.max()
            beta[t, i] = max_v + np.log(np.exp(vals - max_v).sum())

    return beta


def state_posteriors(alpha, beta):
    """
    Compute P(tag_t | obs) for all t using alpha * beta (log-space).

    Returns
    -------
    gamma : (T, N) normalised state posteriors
    """
    log_gamma = alpha + beta
    # Normalise each time step
    max_v = log_gamma.max(axis=1, keepdims=True)
    log_norm = max_v + np.log(np.exp(log_gamma - max_v).sum(axis=1, keepdims=True))
    gamma = np.exp(log_gamma - log_norm)
    return gamma


# ─── Viterbi Algorithm ───────────────────────────────────────────────────────

def viterbi(obs, tag2i, word2i, log_pi, log_A, log_B):
    """
    Decode the most-probable tag sequence using the Viterbi algorithm.

    Returns
    -------
    best_path : list of tag indices
    """
    N = log_pi.shape[0]
    T = len(obs)
    vit   = np.full((T, N), LOG_ZERO, dtype=float)
    back  = np.zeros((T, N), dtype=int)

    w0 = word2i.get(obs[0], -1)
    b0 = log_B[:, w0] if w0 >= 0 else np.zeros(N)
    vit[0] = log_pi + b0

    for t in range(1, T):
        wt = word2i.get(obs[t], -1)
        bt = log_B[:, wt] if wt >= 0 else np.zeros(N)
        scores = vit[t - 1][:, None] + log_A     # (N, N)
        back[t]  = scores.argmax(axis=0)
        vit[t]   = scores.max(axis=0) + bt

    # Backtrack
    path = [int(vit[T - 1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path


# ─── Baum-Welch (simplified EM, 2 iterations) ────────────────────────────────

def baum_welch(obs_seqs, tags, word2i, log_pi, log_A, log_B, n_iter=2):
    """
    Run simplified Baum-Welch EM to re-estimate HMM parameters from
    unlabelled observation sequences.

    Returns updated (log_pi, log_A, log_B) and a list of avg log-probs per iter.
    """
    N = len(tags)
    V = log_B.shape[1]
    iter_log_probs = []

    for iteration in range(n_iter):
        # Accumulators
        pi_acc = np.zeros(N) + 1e-10
        A_acc  = np.zeros((N, N)) + 1e-10
        B_acc  = np.zeros((N, V)) + 1e-10
        total_lp = 0.0
        n_valid  = 0

        for obs in obs_seqs:
            if len(obs) < 2:
                continue
            try:
                lp, alpha = forward_log_prob(obs, None, word2i, log_pi, log_A, log_B)
                beta = backward_log(obs, word2i, log_A, log_B)
            except Exception:
                continue

            if not np.isfinite(lp):
                continue

            T = len(obs)
            gamma = state_posteriors(alpha, beta)  # (T, N)

            # xi: (T-1, N, N) – pairwise posteriors
            for t in range(T - 1):
                wt1 = word2i.get(obs[t + 1], -1)
                bt1 = log_B[:, wt1] if wt1 >= 0 else np.zeros(N)
                log_xi_t = (alpha[t][:, None] + log_A +
                            bt1[None, :] + beta[t + 1][None, :])
                # Normalise
                max_v = log_xi_t.max()
                xi_t = np.exp(log_xi_t - max_v)
                xi_t /= xi_t.sum()
                A_acc += xi_t

            # Accumulate pi and B
            pi_acc += gamma[0]
            for t in range(T):
                wt = word2i.get(obs[t], -1)
                if wt >= 0:
                    B_acc[:, wt] += gamma[t]

            total_lp += lp
            n_valid  += 1

        avg_lp = total_lp / max(n_valid, 1)
        iter_log_probs.append(avg_lp)

        # Re-estimate parameters
        log_pi = np.log(pi_acc / pi_acc.sum())
        log_A  = np.log(A_acc  / A_acc.sum(axis=1, keepdims=True))
        log_B  = np.log(B_acc  / B_acc.sum(axis=1, keepdims=True))

        print(f"    Baum-Welch iter {iteration+1}/{n_iter}  avg_log_prob={avg_lp:.4f}")

    return log_pi, log_A, log_B, iter_log_probs


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_viterbi(test_sents, tags, tag2i, word2i, log_pi, log_A, log_B,
                     sample=200):
    """Compute per-tag and overall Viterbi accuracy on up to `sample` sentences."""
    sents = test_sents[:sample]
    correct_per_tag = collections.defaultdict(int)
    total_per_tag   = collections.defaultdict(int)

    for sent in sents:
        words   = [w for w, _ in sent]
        true_ti = [tag2i.get(t, 0) for _, t in sent]
        pred_ti = viterbi(words, tag2i, word2i, log_pi, log_A, log_B)

        for true, pred in zip(true_ti, pred_ti):
            tag = tags[true]
            total_per_tag[tag]   += 1
            if true == pred:
                correct_per_tag[tag] += 1

    per_tag_acc = {t: correct_per_tag[t] / max(total_per_tag[t], 1)
                   for t in tags if total_per_tag[t] > 0}
    overall = (sum(correct_per_tag.values()) /
               max(sum(total_per_tag.values()), 1))
    return overall, per_tag_acc


# ─── Plotting Functions ───────────────────────────────────────────────────────

def plot_tag_distribution(train_sents):
    tag_counts = collections.Counter(t for s in train_sents for _, t in s)
    top = tag_counts.most_common(20)
    labels, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, counts, color=sns.color_palette('tab20', len(labels)))
    ax.set_title('Top 20 POS Tag Frequencies (Brown Corpus)', fontsize=13)
    ax.set_xlabel('POS Tag')
    ax.set_ylabel('Count')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(c), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cat10_tag_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: cat10_tag_distribution.png")


def plot_transition_heatmap(log_A, tags, top_n=15):
    # Pick the top_n most-common tags by their total outgoing count
    top_tags = tags[:top_n]
    idx = [tags.index(t) for t in top_tags]
    sub = np.exp(log_A[np.ix_(idx, idx)])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(sub, xticklabels=top_tags, yticklabels=top_tags,
                cmap='YlOrRd', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 8})
    ax.set_title('HMM Transition Matrix (top 15 tags)', fontsize=12)
    ax.set_xlabel('Next Tag')
    ax.set_ylabel('Current Tag')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cat10_transition_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: cat10_transition_heatmap.png")


def plot_emission_heatmap(log_B, tags, vocab, top_tags_n=12, top_words_n=20):
    # Select top tags and top words by emission mass
    tag_mass  = np.exp(log_B).sum(axis=1)
    word_mass = np.exp(log_B).sum(axis=0)
    tag_idx   = np.argsort(-tag_mass)[:top_tags_n]
    word_idx  = np.argsort(-word_mass)[:top_words_n]

    sub        = np.exp(log_B[np.ix_(tag_idx, word_idx)])
    row_labels = [tags[i]  for i in tag_idx]
    col_labels = [vocab[i] for i in word_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(sub, xticklabels=col_labels, yticklabels=row_labels,
                cmap='Blues', ax=ax, annot=True, fmt='.3f', annot_kws={'size': 7})
    ax.set_title('HMM Emission Probabilities (top tags × top words)', fontsize=12)
    ax.set_xlabel('Word')
    ax.set_ylabel('POS Tag')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cat10_emission_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: cat10_emission_heatmap.png")


def plot_viterbi_accuracy(per_tag_acc):
    tags_sorted = sorted(per_tag_acc.keys(), key=lambda t: -per_tag_acc[t])
    accs = [per_tag_acc[t] for t in tags_sorted]
    colors = ['#2ecc71' if a >= 0.8 else '#e67e22' if a >= 0.5 else '#e74c3c'
              for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tags_sorted, accs, color=colors)
    ax.axhline(y=sum(accs)/len(accs), color='navy', linestyle='--',
               linewidth=1.5, label=f'Mean = {sum(accs)/len(accs):.3f}')
    ax.set_title('Viterbi POS Tagging Accuracy per Tag', fontsize=12)
    ax.set_xlabel('POS Tag')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{a:.2f}', ha='center', va='bottom', fontsize=9)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cat10_viterbi_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: cat10_viterbi_accuracy.png")


def plot_forward_log_probs(log_probs_list, labels):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette('viridis', len(log_probs_list))
    ax.bar(range(len(log_probs_list)), log_probs_list, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_title('Forward Algorithm: Log P(sentence | HMM model)', fontsize=12)
    ax.set_xlabel('Test Sentence')
    ax.set_ylabel('Log Probability')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cat10_forward_log_probs.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: cat10_forward_log_probs.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CATEGORY 10: HMM POS TAGGING")
    print("=" * 60)

    download_data()

    # ── Load corpus ──────────────────────────────────────────────────────────
    print("\n[Loading Brown Corpus...]")
    all_sents = load_corpus(N_SENTENCES)
    train_sents, test_sents = train_test_split(all_sents)
    print(f"  Total sentences : {len(all_sents)}")
    print(f"  Train sentences : {len(train_sents)}")
    print(f"  Test  sentences : {len(test_sents)}")

    # ── Build HMM ────────────────────────────────────────────────────────────
    print("\n[Estimating HMM parameters (Laplace smoothing)...]")
    tags, vocab, tag2i, word2i, log_pi, log_A, log_B = build_hmm(train_sents)
    N = len(tags)
    V = len(vocab)
    print(f"  Tags  (N) : {N}  -> {tags}")
    print(f"  Vocab (V) : {V}")

    # ── Tag distribution plot ─────────────────────────────────────────────────
    print("\n[Plotting tag distribution...]")
    plot_tag_distribution(train_sents)

    # ── Transition heatmap ───────────────────────────────────────────────────
    print("\n[Plotting transition heatmap...]")
    plot_transition_heatmap(log_A, tags, top_n=min(15, N))

    # ── Emission heatmap ─────────────────────────────────────────────────────
    print("\n[Plotting emission heatmap...]")
    plot_emission_heatmap(log_B, tags, vocab,
                          top_tags_n=min(12, N),
                          top_words_n=20)

    # ── Forward Algorithm ────────────────────────────────────────────────────
    print("\n[Forward Algorithm: computing log P(sentence | model)...]")
    sample_sents = test_sents[:15]
    fwd_log_probs = []
    fwd_labels    = []
    for i, sent in enumerate(sample_sents):
        words = [w for w, _ in sent]
        lp, _ = forward_log_prob(words, tag2i, word2i, log_pi, log_A, log_B)
        fwd_log_probs.append(float(lp))
        fwd_labels.append(f"S{i+1}")
        print(f"  Sentence {i+1:2d} (len={len(words):2d}): log P = {lp:.4f}")

    plot_forward_log_probs(fwd_log_probs, fwd_labels)

    # ── Backward Algorithm (demo: state posteriors on one sentence) ───────────
    print("\n[Backward Algorithm: state posteriors for one sentence...]")
    demo_sent  = test_sents[0]
    demo_words = [w for w, _ in demo_sent]
    demo_true  = [t for _, t in demo_sent]
    _, alpha = forward_log_prob(demo_words, tag2i, word2i, log_pi, log_A, log_B)
    beta      = backward_log(demo_words, word2i, log_A, log_B)
    gamma     = state_posteriors(alpha, beta)   # (T, N)
    print(f"  Sentence: {' '.join(demo_words[:8])}...")
    print(f"  True tags:      {demo_true[:8]}")
    gamma_pred = [tags[int(gamma[t].argmax())] for t in range(len(demo_words))]
    print(f"  Posterior tags: {gamma_pred[:8]}")

    # ── Viterbi Tagging ───────────────────────────────────────────────────────
    print("\n[Viterbi: evaluating on test set (up to 200 sentences)...]")
    overall_acc, per_tag_acc = evaluate_viterbi(
        test_sents, tags, tag2i, word2i, log_pi, log_A, log_B, sample=200
    )
    print(f"  Overall accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    for t in sorted(per_tag_acc, key=lambda x: -per_tag_acc[x]):
        print(f"    {t:<8s}: {per_tag_acc[t]:.4f}")

    plot_viterbi_accuracy(per_tag_acc)

    # ── Baum-Welch EM ─────────────────────────────────────────────────────────
    print("\n[Baum-Welch: re-estimating parameters on unlabelled test data...]")
    # Use word sequences only (strip tags) from test set
    unlabelled_obs = [[w for w, _ in s] for s in test_sents[:150]]

    # Snapshot parameters before BW
    A_before = np.exp(log_A[:3, :3]).copy()
    pi_before = np.exp(log_pi).copy()

    log_pi_bw, log_A_bw, log_B_bw, bw_iter_lps = baum_welch(
        unlabelled_obs, tags, word2i, log_pi.copy(), log_A.copy(), log_B.copy(),
        n_iter=2
    )

    A_after  = np.exp(log_A_bw[:3, :3])
    pi_after = np.exp(log_pi_bw)

    print("\n  π (initial probs) before vs after Baum-Welch:")
    for i, tag in enumerate(tags):
        print(f"    {tag:<8s}: {pi_before[i]:.4f} → {pi_after[i]:.4f}")

    print("\n  Top-left 3×3 of A before vs after:")
    print(f"    Before:\n{A_before}")
    print(f"    After:\n{A_after}")

    # ── Save Metrics ──────────────────────────────────────────────────────────
    metrics = {
        'num_tags'             : N,
        'vocab_size'           : V,
        'train_sentences'      : len(train_sents),
        'test_sentences'       : len(test_sents),
        'viterbi_overall_acc'  : round(overall_acc, 4),
        'viterbi_per_tag_acc'  : {t: round(v, 4) for t, v in per_tag_acc.items()},
        'forward_log_probs'    : {f'S{i+1}': round(lp, 4) for i, lp in
                                  enumerate(fwd_log_probs)},
        'baum_welch_iterations': 2,
        'baum_welch_avg_log_probs': [round(lp, 4) for lp in bw_iter_lps],
        'tags'                 : tags,
    }
    save_metrics(metrics, os.path.join(METRICS_DIR, 'cat10_hmm_pos_tagging.json'))
    print("\n  Metrics saved to outputs/metrics/cat10_hmm_pos_tagging.json")
    print("\n[CATEGORY 10 COMPLETE]")
    return metrics


if __name__ == '__main__':
    run()
