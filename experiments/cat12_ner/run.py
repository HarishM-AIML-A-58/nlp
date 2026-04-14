"""
Category 12: Named Entity Recognition (NER)
============================================
Approach 1 — Rule-Based NER   : gazetteers + contextual rules
Approach 2 — NLTK ne_chunk    : averaged_perceptron_tagger + ne_chunk
Approach 3 — Regex patterns   : money, dates, percentages, emails/URLs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nltk

from utils.evaluation import save_metrics

# ─── Output directories ───────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ─── NLTK downloads ───────────────────────────────────────────────────────────
def _download_nltk_data():
    for res in [
        'brown', 'reuters', 'punkt', 'punkt_tab',
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words',
    ]:
        nltk.download(res, quiet=True)


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Rule-Based NER (gazetteers + contextual rules)
# ═══════════════════════════════════════════════════════════════════════════════

FIRST_NAMES = {
    'james','john','robert','michael','william','david','richard','joseph',
    'thomas','charles','christopher','daniel','matthew','anthony','mark',
    'donald','steven','paul','andrew','kenneth','george','joshua','kevin',
    'brian','edward','ronald','timothy','jason','jeffrey','gary','ryan',
    'jacob','nicholas','eric','jonathan','stephen','larry','justin','scott',
    'brandon','benjamin','samuel','raymond','frank','gregory','raymond',
    'frank','alexander','patrick','jack','dennis','jerry','tyler','aaron',
    'jose','adam','henry','douglas','nathan','peter','zachary','walter',
    'mary','patricia','jennifer','linda','barbara','elizabeth','susan',
    'jessica','sarah','karen','lisa','nancy','betty','margaret','sandra',
    'ashley','dorothy','kimberly','emily','donna','michelle','carol',
    'amanda','melissa','deborah','stephanie','rebecca','sharon','laura',
    'cynthia','kathleen','amy','angela','shirley','anna','brenda','pamela',
    'emma','nicole','helen','samantha','katherine','christine','debra',
}

LAST_NAMES = {
    'smith','johnson','williams','brown','jones','garcia','miller','davis',
    'rodriguez','martinez','hernandez','lopez','gonzalez','wilson','anderson',
    'thomas','taylor','moore','jackson','martin','lee','perez','thompson',
    'white','harris','sanchez','clark','ramirez','lewis','robinson','walker',
    'young','allen','king','wright','scott','torres','nguyen','hill','flores',
    'green','adams','nelson','baker','hall','rivera','campbell','mitchell',
    'carter','roberts','turner','phillips','evans','collins','edwards',
    'stewart','morris','morgan','reed','cook','bell','murphy','bailey',
    'cooper','richardson','cox','howard','ward','torres','peterson','gray',
    'james','watson','brooks','kelly','sanders','price','bennett','wood',
    'barnes','ross','henderson','coleman','jenkins','perry','powell','long',
    'patterson','hughes','flores','washington','butler','simmons','foster',
    'gonzales','bryant','alexander','russell','griffin','diaz','hayes',
}

ORGANIZATIONS = {
    'microsoft','google','ibm','apple','amazon','facebook','meta','twitter',
    'netflix','tesla','intel','oracle','cisco','adobe','salesforce','uber',
    'airbnb','linkedin','spotify','snapchat','samsung','sony','toyota',
    'volkswagen','bmw','ford','general motors','general electric','boeing',
    'lockheed','raytheon','exxon','chevron','shell','bp','halliburton',
    'goldman sachs','jpmorgan','morgan stanley','citigroup','bank of america',
    'wells fargo','hsbc','barclays','deutsche bank','credit suisse',
    'un','united nations','nato','who','imf','world bank','unicef',
    'red cross','greenpeace','amnesty international','oxfam',
    'harvard','mit','stanford','oxford','cambridge','yale','princeton',
    'nfl','nba','fifa','ioc','espn','bbc','cnn','nbc','abc','cbs','fox',
    'new york times','washington post','wall street journal','reuters',
    'associated press','bloomberg','the guardian','the times',
    'nasa','fbi','cia','nsa','dhs','doj','sec','fda','epa','pentagon',
    'white house','congress','senate','supreme court',
    'aclu','naacp','aarp','afl-cio','chamber of commerce',
}

COUNTRIES = {
    'afghanistan','albania','algeria','andorra','angola','argentina',
    'armenia','australia','austria','azerbaijan','bahamas','bahrain',
    'bangladesh','barbados','belarus','belgium','belize','benin','bhutan',
    'bolivia','bosnia','botswana','brazil','brunei','bulgaria','burkina faso',
    'burundi','cambodia','cameroon','canada','chad','chile','china','colombia',
    'comoros','congo','costa rica','croatia','cuba','cyprus','czechia',
    'denmark','djibouti','dominica','ecuador','egypt','eritrea','estonia',
    'ethiopia','fiji','finland','france','gabon','gambia','georgia','germany',
    'ghana','greece','grenada','guatemala','guinea','guyana','haiti',
    'honduras','hungary','iceland','india','indonesia','iran','iraq','ireland',
    'israel','italy','jamaica','japan','jordan','kazakhstan','kenya','kuwait',
    'kyrgyzstan','laos','latvia','lebanon','lesotho','liberia','libya',
    'liechtenstein','lithuania','luxembourg','madagascar','malawi','malaysia',
    'maldives','mali','malta','mauritania','mauritius','mexico','moldova',
    'monaco','mongolia','montenegro','morocco','mozambique','myanmar',
    'namibia','nepal','netherlands','new zealand','nicaragua','niger',
    'nigeria','norway','oman','pakistan','panama','paraguay','peru',
    'philippines','poland','portugal','qatar','romania','russia','rwanda',
    'saudi arabia','senegal','serbia','singapore','slovakia','slovenia',
    'somalia','south africa','south korea','spain','sri lanka','sudan',
    'sweden','switzerland','syria','taiwan','tajikistan','tanzania','thailand',
    'togo','trinidad','tunisia','turkey','turkmenistan','uganda','ukraine',
    'united arab emirates','united kingdom','united states','uruguay',
    'uzbekistan','venezuela','vietnam','yemen','zambia','zimbabwe',
    'uk','usa','uae','ussr','eu',
}

CITIES = {
    'new york','los angeles','chicago','houston','phoenix','philadelphia',
    'san antonio','san diego','dallas','san jose','austin','jacksonville',
    'fort worth','columbus','charlotte','san francisco','indianapolis',
    'seattle','denver','nashville','washington','boston','el paso','detroit',
    'memphis','portland','las vegas','louisville','baltimore','milwaukee',
    'albuquerque','tucson','fresno','mesa','sacramento','atlanta','kansas city',
    'omaha','colorado springs','raleigh','miami','virginia beach','oakland',
    'minneapolis','tulsa','tampa','arlington','new orleans',
    'london','paris','berlin','madrid','rome','amsterdam','brussels',
    'vienna','zurich','stockholm','oslo','copenhagen','helsinki',
    'moscow','beijing','shanghai','tokyo','seoul','delhi','mumbai',
    'cairo','johannesburg','nairobi','sydney','melbourne','toronto',
    'montreal','mexico city','sao paulo','buenos aires','lima','bogota',
    'lagos','accra','dubai','istanbul','tehran','riyadh','singapore',
    'hong kong','taipei','bangkok','jakarta','manila','dhaka','karachi',
}

US_STATES = {
    'alabama','alaska','arizona','arkansas','california','colorado',
    'connecticut','delaware','florida','georgia','hawaii','idaho','illinois',
    'indiana','iowa','kansas','kentucky','louisiana','maine','maryland',
    'massachusetts','michigan','minnesota','mississippi','missouri','montana',
    'nebraska','nevada','new hampshire','new jersey','new mexico','new york',
    'north carolina','north dakota','ohio','oklahoma','oregon','pennsylvania',
    'rhode island','south carolina','south dakota','tennessee','texas','utah',
    'vermont','virginia','washington','west virginia','wisconsin','wyoming',
}

ALL_LOCATIONS = COUNTRIES | CITIES | US_STATES

TITLE_TOKENS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'sen.', 'rep.', 'gov.'}


def rule_based_ner(tokens):
    """
    Return list of (entity_text, entity_type) tuples using gazetteers and
    contextual title rules.
    """
    entities = []
    i = 0
    lower_tokens = [t.lower() for t in tokens]

    while i < len(tokens):
        # Title rule: title token followed by a capitalised word → PERSON
        if lower_tokens[i] in TITLE_TOKENS and i + 1 < len(tokens) and tokens[i + 1][0].isupper():
            name = tokens[i + 1]
            # try to grab a second capitalised word (last name)
            if i + 2 < len(tokens) and tokens[i + 2][0].isupper() and tokens[i + 2].isalpha():
                name = name + ' ' + tokens[i + 2]
                i += 3
            else:
                i += 2
            entities.append((name, 'PERSON'))
            continue

        # Try 2-word look-ups first (multi-word entities)
        if i + 1 < len(tokens):
            bigram = lower_tokens[i] + ' ' + lower_tokens[i + 1]
            if bigram in ORGANIZATIONS:
                entities.append((tokens[i] + ' ' + tokens[i + 1], 'ORG'))
                i += 2
                continue
            if bigram in ALL_LOCATIONS:
                entities.append((tokens[i] + ' ' + tokens[i + 1], 'LOC'))
                i += 2
                continue

        tok_l = lower_tokens[i]

        if tok_l in ORGANIZATIONS:
            entities.append((tokens[i], 'ORG'))
            i += 1
            continue

        if tok_l in ALL_LOCATIONS:
            entities.append((tokens[i], 'LOC'))
            i += 1
            continue

        # Person: capitalised word whose lower form is in first_names or last_names
        if tokens[i][0].isupper() and tok_l in (FIRST_NAMES | LAST_NAMES):
            name = tokens[i]
            if (i + 1 < len(tokens) and tokens[i + 1][0].isupper()
                    and lower_tokens[i + 1] in LAST_NAMES):
                name = name + ' ' + tokens[i + 1]
                i += 2
            else:
                i += 1
            entities.append((name, 'PERSON'))
            continue

        i += 1

    return entities


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — NLTK ne_chunk
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_entities(tagged_sent):
    """Extract (entity_text, entity_type) from an ne_chunk tree."""
    entities = []
    try:
        chunked = nltk.ne_chunk(tagged_sent, binary=False)
    except Exception:
        return entities
    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            etype = subtree.label()
            etext = ' '.join(w for w, _ in subtree.leaves())
            # Map GPE → LOC for consistency
            if etype == 'GPE':
                etype = 'LOC'
            elif etype not in ('PERSON', 'ORGANIZATION'):
                etype = 'MISC'
            if etype != 'MISC':
                entities.append((etext, etype if etype != 'ORGANIZATION' else 'ORG'))
    return entities


def nltk_ner_on_sents(tagged_sents):
    """Run ne_chunk on a list of already-POS-tagged sentences."""
    all_entities = []
    for ts in tagged_sents:
        all_entities.extend(_chunk_entities(ts))
    return all_entities


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 3 — Regex-based patterns
# ═══════════════════════════════════════════════════════════════════════════════

REGEX_PATTERNS = {
    'MONEY':      re.compile(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?'),
    'DATE':       re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2}(?:,\s*\d{4})?'
        r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        r'|\b\d{4}-\d{2}-\d{2}\b'
    ),
    'PERCENT':    re.compile(r'\b\d+(?:\.\d+)?%'),
    'EMAIL':      re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'),
    'URL':        re.compile(r'https?://[^\s]+'),
    'PHONE':      re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'),
}


def regex_ner(text):
    entities = []
    for etype, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            entities.append((m.group(), etype))
    return entities


# ═══════════════════════════════════════════════════════════════════════════════
# GOLD EVALUATION SET (hand-labelled)
# ═══════════════════════════════════════════════════════════════════════════════

GOLD_SENTENCES = [
    ("Barack Obama was born in Hawaii and later became president of the United States.",
     [('Barack Obama', 'PERSON'), ('Hawaii', 'LOC'), ('United States', 'LOC')]),
    ("Microsoft and Google are competing for dominance in the cloud computing market.",
     [('Microsoft', 'ORG'), ('Google', 'ORG')]),
    ("Dr. Angela Merkel served as Chancellor of Germany for sixteen years.",
     [('Angela Merkel', 'PERSON'), ('Germany', 'LOC')]),
    ("The United Nations held an emergency session in New York to address the crisis.",
     [('United Nations', 'ORG'), ('New York', 'LOC')]),
    ("Elon Musk founded Tesla and SpaceX after selling PayPal to eBay.",
     [('Elon Musk', 'PERSON'), ('Tesla', 'ORG'), ('SpaceX', 'ORG'), ('PayPal', 'ORG'), ('eBay', 'ORG')]),
    ("Amazon opened a new distribution centre in Seattle last month.",
     [('Amazon', 'ORG'), ('Seattle', 'LOC')]),
    ("Mr. James Brown won the Nobel Prize in Physics.",
     [('James Brown', 'PERSON')]),
    ("The French government announced new measures affecting Paris and Lyon.",
     [('Paris', 'LOC'), ('Lyon', 'LOC')]),
    ("IBM researchers published a breakthrough paper on quantum computing.",
     [('IBM', 'ORG')]),
    ("Prof. Stephen Hawking worked at the University of Cambridge for decades.",
     [('Stephen Hawking', 'PERSON'), ('Cambridge', 'LOC')]),
    ("Apple released its new iPhone model at the Steve Jobs Theater in Cupertino.",
     [('Apple', 'ORG'), ('Steve Jobs', 'PERSON'), ('Cupertino', 'LOC')]),
    ("The Red Cross delivered aid to refugees in Syria and Lebanon.",
     [('Red Cross', 'ORG'), ('Syria', 'LOC'), ('Lebanon', 'LOC')]),
    ("Senator John McCain represented Arizona for many years.",
     [('John McCain', 'PERSON'), ('Arizona', 'LOC')]),
    ("Netflix produced award-winning content in Los Angeles and Toronto.",
     [('Netflix', 'ORG'), ('Los Angeles', 'LOC'), ('Toronto', 'LOC')]),
    ("Mrs. Elizabeth Warren delivered a speech about economic inequality.",
     [('Elizabeth Warren', 'PERSON')]),
    ("Toyota and Ford announced a joint venture to produce electric vehicles.",
     [('Toyota', 'ORG'), ('Ford', 'ORG')]),
    ("The World Bank released its annual report on poverty in Africa.",
     [('World Bank', 'ORG'), ('Africa', 'LOC')]),
    ("David Beckham played for Manchester United and Real Madrid.",
     [('David Beckham', 'PERSON')]),
    ("NASA launched a new mission to explore Mars and the outer solar system.",
     [('NASA', 'ORG'), ('Mars', 'LOC')]),
    ("The European Union imposed sanctions on Russia following the invasion of Ukraine.",
     [('European Union', 'ORG'), ('Russia', 'LOC'), ('Ukraine', 'LOC')]),
    ("Bill Gates and Warren Buffett pledged billions to charity through the Giving Pledge.",
     [('Bill Gates', 'PERSON'), ('Warren Buffett', 'PERSON')]),
    ("Sony and Samsung dominate the consumer electronics market in Asia.",
     [('Sony', 'ORG'), ('Samsung', 'ORG'), ('Asia', 'LOC')]),
    ("The BBC reported live from London during the royal ceremony.",
     [('BBC', 'ORG'), ('London', 'LOC')]),
    ("Dr. Martin Luther King Jr. led the civil rights movement in the United States.",
     [('Martin Luther King', 'PERSON'), ('United States', 'LOC')]),
    ("Intel unveiled its latest processor at a conference in San Francisco.",
     [('Intel', 'ORG'), ('San Francisco', 'LOC')]),
    ("Amnesty International criticized the government of China for human rights abuses.",
     [('Amnesty International', 'ORG'), ('China', 'LOC')]),
    ("The Tokyo Olympics were delayed by one year due to the pandemic.",
     [('Tokyo', 'LOC')]),
    ("Goldman Sachs reported record profits despite a turbulent economic climate.",
     [('Goldman Sachs', 'ORG')]),
    ("Rep. Nancy Pelosi announced new legislation targeting pharmaceutical companies.",
     [('Nancy Pelosi', 'PERSON')]),
    ("Facebook and Twitter faced scrutiny from the US Senate over data privacy.",
     [('Facebook', 'ORG'), ('Twitter', 'ORG'), ('Senate', 'ORG')]),
]


def _evaluate_ner(predicted_entities, gold_entities):
    """
    Compute per-type TP, FP, FN given lists of (text, type) tuples.
    Matching is case-insensitive on text and exact on type.
    """
    types = ['PERSON', 'ORG', 'LOC']
    counts = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in types}

    for pred_text, pred_type in predicted_entities:
        if pred_type not in types:
            continue
        # A prediction is TP if an identical (text, type) exists in gold
        matched = any(
            pred_text.lower() == g.lower() and pred_type == gt
            for g, gt in gold_entities
        )
        if matched:
            counts[pred_type]['tp'] += 1
        else:
            counts[pred_type]['fp'] += 1

    for gold_text, gold_type in gold_entities:
        if gold_type not in types:
            continue
        found = any(
            gold_text.lower() == p.lower() and gold_type == pt
            for p, pt in predicted_entities
        )
        if not found:
            counts[gold_type]['fn'] += 1

    results = {}
    for t in types:
        tp, fp, fn = counts[t]['tp'], counts[t]['fp'], counts[t]['fn']
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        results[t] = {'precision': round(prec, 4),
                      'recall':    round(rec,  4),
                      'f1':        round(f1,   4),
                      'tp': tp, 'fp': fp, 'fn': fn}
    return results


def run_evaluation():
    """
    Evaluate both rule-based NER and NLTK ne_chunk on the gold set.
    Returns dict with evaluation results.
    """
    rule_all_pred, rule_all_gold = [], []
    nltk_all_pred, nltk_all_gold = [], []

    for sentence, gold in GOLD_SENTENCES:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        # Rule-based
        rb_entities = rule_based_ner(tokens)
        rule_all_pred.extend(rb_entities)
        rule_all_gold.extend(gold)

        # NLTK ne_chunk
        nc_entities = _chunk_entities(tagged)
        nltk_all_pred.extend(nc_entities)
        nltk_all_gold.extend(gold)

    rule_eval = _evaluate_ner(rule_all_pred, rule_all_gold)
    nltk_eval = _evaluate_ner(nltk_all_pred, nltk_all_gold)

    return {'rule_based': rule_eval, 'nltk_ne_chunk': nltk_eval}


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS ANALYSIS — Brown + Reuters
# ═══════════════════════════════════════════════════════════════════════════════

def corpus_ner_analysis(max_sents=200):
    from nltk.corpus import brown, reuters

    person_counts  = collections.Counter()
    org_counts     = collections.Counter()
    loc_counts     = collections.Counter()
    entity_lengths = []

    # Brown has tagged_sents; Reuters is plaintext only — use sents()
    brown_sents   = [list(s) for s in brown.sents()[:max_sents]]
    reuters_sents = [list(s) for s in reuters.sents()[:max_sents]]

    corpora = [
        ('brown',   brown_sents),
        ('reuters', reuters_sents),
    ]

    for corpus_name, sent_list in corpora:
        print(f"  Processing {corpus_name} corpus ({len(sent_list)} sentences)…")
        # Batch POS-tag all sentences at once — much faster than one-by-one
        try:
            all_tagged = nltk.pos_tag_sents(sent_list)
        except Exception:
            all_tagged = [nltk.pos_tag(s) for s in sent_list]

        for tagged_sent in all_tagged:
            try:
                entities = _chunk_entities(tagged_sent)
            except Exception:
                continue

            for etext, etype in entities:
                length = len(etext.split())
                entity_lengths.append(length)
                if etype == 'PERSON':
                    person_counts[etext.title()] += 1
                elif etype == 'ORG':
                    org_counts[etext.title()] += 1
                elif etype == 'LOC':
                    loc_counts[etext.title()] += 1

    return person_counts, org_counts, loc_counts, entity_lengths


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _bar_top_n(counter, title, filename, n=20, color='steelblue'):
    if not counter:
        return
    top = counter.most_common(n)
    labels, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(labels)), vals, color=color, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100)
    plt.close()


def plot_entity_distribution(person_cnt, org_cnt, loc_cnt):
    totals = {
        'PERSON': sum(person_cnt.values()),
        'ORG':    sum(org_cnt.values()),
        'LOC':    sum(loc_cnt.values()),
    }
    labels  = list(totals.keys())
    values  = list(totals.values())
    colors  = ['#4e79a7', '#f28e2b', '#59a14f']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axes[0].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    axes[0].set_title('Entity Type Distribution (Pie)')

    # Bar chart
    axes[1].bar(labels, values, color=colors, edgecolor='white')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Entity Type Distribution (Bar)')
    for i, v in enumerate(values):
        axes[1].text(i, v + max(values) * 0.01, str(v), ha='center', fontsize=11)

    plt.suptitle('Named Entity Type Distribution — Brown + Reuters Corpora', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat12_entity_type_distribution.png'), dpi=100)
    plt.close()


def plot_entity_length(entity_lengths):
    if not entity_lengths:
        entity_lengths = [1]
    max_len = min(max(entity_lengths), 8)
    bins = list(range(1, max_len + 2))
    counts = [entity_lengths.count(l) for l in range(1, max_len + 1)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, max_len + 1), counts, color='mediumpurple', edgecolor='white')
    ax.set_xticks(range(1, max_len + 1))
    ax.set_xlabel('Entity Length (words)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Entity Mention Lengths')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat12_entity_length_dist.png'), dpi=100)
    plt.close()


def plot_precision_recall(eval_results):
    entity_types = ['PERSON', 'ORG', 'LOC']
    approaches   = list(eval_results.keys())  # rule_based, nltk_ne_chunk
    metrics      = ['precision', 'recall', 'f1']
    colors       = ['#4e79a7', '#f28e2b', '#59a14f']

    x   = np.arange(len(entity_types))
    w   = 0.13
    n_m = len(metrics)
    n_a = len(approaches)
    total_groups = n_a * n_m

    fig, ax = plt.subplots(figsize=(14, 6))

    offset_start = -(total_groups - 1) / 2 * w
    idx = 0
    legend_handles = []
    for a_i, approach in enumerate(approaches):
        for m_i, metric in enumerate(metrics):
            vals = [eval_results[approach].get(et, {}).get(metric, 0.0)
                    for et in entity_types]
            offsets = x + offset_start + idx * w
            bar = ax.bar(offsets, vals, w * 0.9,
                         color=colors[m_i],
                         alpha=0.7 + 0.15 * a_i,
                         label=f'{approach} – {metric}',
                         hatch='//' if a_i == 1 else '')
            legend_handles.append(bar)
            idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(entity_types)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('NER Precision / Recall / F1 — Rule-Based vs NLTK ne_chunk')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cat12_ner_precision_recall.png'), dpi=100)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 3 DEMO
# ═══════════════════════════════════════════════════════════════════════════════

REGEX_TEST_TEXTS = [
    "The deal was worth $3.5 billion and was signed on January 15, 2024.",
    "Revenue increased by 12.5% year-over-year, driven by cloud services.",
    "Contact us at info@company.com or visit https://www.example.com.",
    "Call 800-555-1234 to speak with a representative about your account.",
    "The budget was cut from $500,000 to $350,000 between March 2023 and June 2023.",
    "Taxes rose 8% while inflation hit 6.7% in Q3/2023.",
    "The project manager emailed sarah.jones@org.net a report on 2024-03-22.",
]


def run_regex_demo():
    print("\n  Regex-Based NER Demo:")
    print("  " + "-" * 60)
    regex_counts = collections.Counter()
    for text in REGEX_TEST_TEXTS:
        entities = regex_ner(text)
        for _, etype in entities:
            regex_counts[etype] += 1
        print(f"  TEXT: {text[:70]}…" if len(text) > 70 else f"  TEXT: {text}")
        for etext, etype in entities:
            print(f"    [{etype}] {etext}")
    return regex_counts


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 65)
    print("  Category 12 — Named Entity Recognition")
    print("=" * 65)

    _download_nltk_data()

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[1/4] Evaluating Rule-Based and NLTK ne_chunk NER on gold set…")
    eval_results = run_evaluation()

    for approach, per_type in eval_results.items():
        print(f"\n  {approach}:")
        for etype, scores in per_type.items():
            print(f"    {etype:8s}  P={scores['precision']:.3f}  "
                  f"R={scores['recall']:.3f}  F1={scores['f1']:.3f}  "
                  f"(TP={scores['tp']} FP={scores['fp']} FN={scores['fn']})")

    # ── Corpus analysis ───────────────────────────────────────────────────────
    print("\n[2/4] Running NLTK ne_chunk on Brown + Reuters corpora…")
    person_cnt, org_cnt, loc_cnt, ent_lengths = corpus_ner_analysis(max_sents=200)

    total_persons = sum(person_cnt.values())
    total_orgs    = sum(org_cnt.values())
    total_locs    = sum(loc_cnt.values())
    print(f"  PERSON={total_persons}  ORG={total_orgs}  LOC={total_locs}")
    print(f"  Top persons: {person_cnt.most_common(5)}")
    print(f"  Top orgs:    {org_cnt.most_common(5)}")
    print(f"  Top locs:    {loc_cnt.most_common(5)}")

    # ── Regex demo ────────────────────────────────────────────────────────────
    print("\n[3/4] Regex-based NER patterns…")
    regex_counts = run_regex_demo()

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Generating plots…")

    plot_entity_distribution(person_cnt, org_cnt, loc_cnt)
    print("  ✓ cat12_entity_type_distribution.png")

    _bar_top_n(person_cnt, 'Top 20 Person Mentions — Brown + Reuters',
               'cat12_top_persons.png', color='#4e79a7')
    print("  ✓ cat12_top_persons.png")

    _bar_top_n(org_cnt, 'Top 20 Organisation Mentions — Brown + Reuters',
               'cat12_top_organizations.png', color='#f28e2b')
    print("  ✓ cat12_top_organizations.png")

    _bar_top_n(loc_cnt, 'Top 20 Location Mentions — Brown + Reuters',
               'cat12_top_locations.png', color='#59a14f')
    print("  ✓ cat12_top_locations.png")

    plot_precision_recall(eval_results)
    print("  ✓ cat12_ner_precision_recall.png")

    plot_entity_length(ent_lengths)
    print("  ✓ cat12_entity_length_dist.png")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        'total_entities_found': {
            'PERSON': total_persons,
            'ORG':    total_orgs,
            'LOC':    total_locs,
        },
        'per_type_counts': {
            'top_persons':       dict(person_cnt.most_common(20)),
            'top_organizations': dict(org_cnt.most_common(20)),
            'top_locations':     dict(loc_cnt.most_common(20)),
        },
        'evaluation_f1_scores': {
            approach: {et: scores['f1'] for et, scores in per_type.items()}
            for approach, per_type in eval_results.items()
        },
        'evaluation_full': eval_results,
        'corpus_stats': {
            'sentences_per_corpus': 200,
            'avg_entity_length_words': (
                round(sum(ent_lengths) / max(len(ent_lengths), 1), 2)
            ),
            'total_entity_mentions': total_persons + total_orgs + total_locs,
        },
        'regex_pattern_counts': dict(regex_counts),
        'gold_set_sentences': len(GOLD_SENTENCES),
    }

    save_metrics(metrics, os.path.join(METRICS_DIR, 'cat12_ner.json'))
    print("\n  ✓ Metrics saved to outputs/metrics/cat12_ner.json")
    print("\nDone.")


if __name__ == '__main__':
    run()
