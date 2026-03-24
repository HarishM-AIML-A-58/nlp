# Lab Record — Category 3: Rule-Based Morphological Analyzer

## Aim
To implement a rule-based morphological analyzer that identifies morphemes, stem words, and predict POS using hand-crafted rules.

## Objective
1. Define suffix stripping rules for noun, verb, adjective, adverb derivation
2. Define prefix detection rules for negation, temporal, and spatial morphemes
3. Apply POS-based transformation rules (e.g., ADJ→NOUN via -ness)
4. Evaluate on Brown Corpus word sample

## Dataset Description
- **Brown Corpus** (NLTK): 3,000 sentences used; unique word sample of ~1,000 words extracted for analysis
- Custom word list of 13 morphologically complex words for demonstration

## Algorithm

### Suffix Stripping (30 rules covering):
- **Nominalisation**: -tion, -ation, -ment, -ness, -ity, -ance, -ence
- **Agent nouns**: -er, -or, -ist
- **Adjectives**: -ful, -less, -ous, -ious, -al, -ical, -ive, -able, -ible
- **Verbs**: -ing, -ed, -ize, -ise, -fy
- **Adverbs**: -ly

### Prefix Detection (20 rules covering):
- **Negation**: un-, dis-, in-, im-, non-
- **Temporal**: pre-, post-, re-
- **Spatial/relational**: sub-, super-, inter-, trans-, co-
- **Scale**: micro-, macro-, auto-, bio-

### POS Transformation Rules:
```
ADJ(-ful) → NOUN (base)     e.g., helpful → help
NOUN(-ness) → ADJ (base)   e.g., happiness → happy
ADV(-ly) → ADJ (base)      e.g., quickly → quick
VERB(-er) → NOUN (agent)   e.g., teach → teacher
```

### Stepwise Algorithm
1. Convert word to lowercase
2. Check each prefix rule (longest match first)
3. Check each suffix rule (longest match first)
4. Verify minimum stem length (≥3 characters)
5. Assign best-guess POS based on first matching suffix
6. Apply POS transformations where applicable

## Observations
- **Coverage**: ~60–70% of Brown corpus words match at least one rule
- **Prefix frequency**: un- is the most common prefix, followed by dis- and re-
- **Suffix frequency**: -ing and -ed dominate (verb morphology); -ly is the most common adverb marker
- **POS distribution**: VERB and ADJ suffixes most frequent due to rich inflection
- **Error types**: Over-stemming (e.g., "early" → stem "ear" via -ly), under-detection for irregular forms

## Inference
- Rule-based morphology achieves good precision for regular patterns but fails on irregular forms
- Prefix analysis correctly identifies negation morphemes with high accuracy
- Minimum stem length constraint reduces over-analysis significantly
- Rule-based systems are interpretable but require extensive hand-crafting for broad coverage

## Results

| Category | Count (of ~1000 words) | % |
|----------|----------------------|---|
| Words with detected prefix | ~150–200 | 15–20% |
| Words with detected suffix | ~450–550 | 45–55% |
| POS=VERB | ~200–250 | 20–25% |
| POS=NOUN | ~150–200 | 15–20% |
| POS=ADJ  | ~100–150 | 10–15% |
| POS=ADV  | ~80–120  | 8–12% |

**Outputs**: `outputs/plots/cat3_*.png`, `outputs/metrics/cat3_rule_morphology.json`
