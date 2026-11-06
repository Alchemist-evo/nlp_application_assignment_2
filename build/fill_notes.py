"""Insert the analysis markdown into the EXECUTED notebook (markdown cells only; outputs are untouched).
Every number below was copied from the printed outputs of the executed notebook."""
import json

NB = "Assignment2_Programming_Group_86.ipynb"

NOTES = {}

NOTES["A_NOTES"] = r"""### Part A – what we did and what the EDA shows

**Data preparation decisions**
- **Loading.** The installed `datasets` 5.0 cannot map the parquet revision to the named config, so the notebook catches that error (printed above) and reads *the same* `science/{split}/0000.parquet` files of revision `refs/convert/parquet` directly. Split sizes match the brief: 103 / 351 / 400.
- **Training data = train + validation (454 sentences)**, as the brief requires; all results are on the 400-sentence test split. A fixed 10 % slice of train+validation (45 sentences, seed 42) is used **only** for choosing epochs / drawing loss curves in Parts C and D; every final model is retrained on all 454 sentences.
- **Span → BIO.** CrossRE end indices are **inclusive**, so `end + 1` is applied before `spans_to_bio`. Verification: (i) a full **round-trip check** (`bio_to_spans(tags) == gold spans`) gives **0 mismatches in all three splits**; (ii) CrossRE-science has **0 overlapping/nested entities**, so BIO loses nothing; (iii) the three printed sentences show the dataset's inclusive annotation next to the produced tags, e.g. `University of Göttingen` = (6, 8 inclusive) → `B-/I-/I-university`, `Adenosine triphosphate` = (4, 5) → `B-/I-chemicalcompound`.
- **Double relation labels.** When an entity pair carries two relation labels we **keep the first annotated label**; this dropped 46 (train), 47 (validation) and 53 (test) extra labels.

**What the EDA shows**
- Sentences are long (≈36 tokens on average, max 84) and dense: ≈5.6 entities per sentence and ≈34.7 % of all tokens lie inside an entity, so context windows matter.
- Average entity length is **2.23 tokens** (train+val) / 2.15 (test), but it varies strongly by type: `book` 5.56, `award` 4.02, `organisation` 3.46 vs. `astronomicalobject` 1.28 – long types are where **boundary errors** are expected.
- **Class imbalance:** the three biggest types (`astronomicalobject` 14.1 %, `misc` 13.7 %, `scientist` 12.5 %) cover 40 % of mentions, while six types have < 50 training mentions (`book` 6, `theory` 11, `chemicalelement` 25, `country` 27, `discipline` 42, `event` 44).
- **Train/test shift:** `misc` rises from 13.7 % to 17.5 % of mentions in test, `chemicalelement` has 25 training vs 62 test mentions and `protein` 86 vs 125 – the test set is harder than the training distribution suggests.
"""

NOTES["B_NOTES"] = r"""### Part B – feature justification and results

**Features (and why):**
| group | features | why it should help on CrossRE-science |
|---|---|---|
| lexical | lower-cased word, length | memorises frequent entity words (`Royal`, `Nobel`, `Society`) |
| case / shape | title/upper/digit flags, inner capital, full and compressed word shape (`Xxxx`, `Xx-d`) | names are capitalised; formulas/codes (`NaCl`, `CCL5`, `1139 Atami`) have distinctive shapes |
| affix | prefixes (2–3), suffixes (2–4) | morphology carries the type: `-ase` enzymes, `-ium` elements, `-us` astronomical objects, `-ology` disciplines |
| POS | NLTK tag of word and neighbours | proper nouns (NNP) vs. common nouns; numbers before minor-planet names |
| context | words, case, shape and POS at ±1/±2, bigrams | trigger words: *in* → location, *University of* → university, *discovered by* → scientist |
| domain | chemistry suffix list, letter+digit mix, Greek letters | chemical/protein names such as `NF-κB`, `H2A`, `aciclovir` |

**Tuning and ablation (5-fold CV on train+validation, test untouched).** The 45-sentence dev slice was too small for stable CRF decisions, so we used 5-fold CV. Best regularisation: **c1 = 0.1, c2 = 0.1 (CV F1 = 0.5745)**. Removing *any* feature group lowers CV F1, which justifies keeping all six: **affixes matter most (−0.0255)**, then context (−0.0106), case/shape (−0.0086), domain cues (−0.0053), lexical (−0.0035) and POS (−0.0006, nearly redundant with shape/context). The learned top weights confirm this: `-1:lower:in → B-location`, `1:lower:university → B-university`, `suf2:um → B-chemicalelement`, `suf3:nus → B-astronomicalobject`, `1:pos:VBD → B-scientist` (the subject of *discovered*).

**Test results (full 400-sentence test):** micro **P = 0.6117, R = 0.5552, F1 = 0.5821**. The CRF is precise but conservative (P > R). It handles types with regular surface patterns well – `astronomicalobject` 0.8525, `award` 0.8214, `university` 0.8047 – and is **poor on** `book` (0.0, only 3 test mentions), `theory` (0.1538), `country` (0.3200), `protein` (0.3448) and the heterogeneous `misc` (0.3459), i.e. rare types and types whose names have no consistent morphology.
"""

NOTES["C_NOTES"] = r"""### Part C – architecture and comparison with the CRF

**Architecture.** Word embedding initialised from **GloVe-100 (glove-wiki-gigaword-100)** and fine-tuned + **character CNN** (30-d chars, 50 filters, width 3, max-pool) + 8-d casing embedding → dropout 0.5 → **BiLSTM** (2 × 128) → linear emissions → **CRF layer** (`pytorch-crf`) that scores whole tag sequences, so invalid transitions such as `O → I-x` are learned to be unlikely. Adam (lr 1e-3), batch 16, gradient clipping 5. GloVe covers 97.1 % of test tokens but only 93.2 % of test *entity* tokens, which is why the character CNN is added: it builds vectors for unseen chemical and protein names from their spelling.

**Loss curves.** Training loss falls monotonically (77.5 → 2.44 NLL/sentence over 40 epochs). Dev loss falls until about epoch 30 (13.67) and then stops improving while the training loss keeps dropping – the model starts to overfit the 409 training sentences. Dev F1 peaks at **0.6022 at epoch 30**, so the final model is retrained on all 454 sentences for 30 epochs (123 s on CPU).

**Results vs. CRF (full test).** BiLSTM-CRF **F1 = 0.6507** (P 0.6490, R 0.6524) vs. CRF 0.5821, i.e. **+0.069 F1**, mostly from recall (0.6524 vs 0.5552): dense embeddings generalise to words never seen in training. Largest gains: `person` 0.42 → 0.72, `location` 0.43 → 0.71, `scientist` 0.70 → 0.85. It is **worse** than the CRF on rare chemistry types – `chemicalelement` 0.5625 → 0.0588 and `enzyme` 0.6019 → 0.4318 – where the CRF's explicit suffix features (`-um`, `-ase`) are more data-efficient than learned representations from 25–52 examples.
"""

NOTES["D_NOTES"] = r"""### Part D – alignment, training settings and results

**Subword-to-word alignment (shown in the table above).** WordPiece splits words into sub-tokens, but labels exist per *word*. Using the fast tokenizer's `word_ids()`, the **first sub-token of each word receives the word's label** and every continuation sub-token, plus `[CLS]`/`[SEP]`, receives **-100**, which PyTorch's cross-entropy ignores. In the example, `Adenosine` → `Aden` (B-chemicalcompound) + `##osine` (-100) and `triphosphate` → `trip` (I-chemicalcompound) + `##ho ##sp ##hat ##e` (all -100). Therefore each word contributes exactly one loss term regardless of how many pieces it is split into. **At prediction time the same mapping is reversed:** the label predicted at each word's first sub-token becomes the word's tag. The result is a word-level sequence of exactly the gold length, scored with seqeval like every other model; a small `fix_bio` step turns an orphan `I-x` into `B-x`. Important detail: the SciBERT-*cased* checkpoint has no tokenizer config, and by default it would lower-case the input, so `do_lower_case=False` is set explicitly (visible in the example: `Aden`, not `aden`). The longest sentence has 141 sub-tokens, so nothing is truncated.

**Training settings.** `allenai/scibert_scivocab_cased`, AdamW lr 5e-5, linear decay with 10 % warm-up, weight decay 0.01, batch 16, seed 42. On the fit/dev split, dev F1 rose to **0.6778 at epoch 9** (then 0.66 at epochs 10–12). Dev *loss* was lowest at epoch 4 (0.4844), but it rises afterwards because the model becomes over-confident on a few errors while its entity decisions keep improving, so the epoch is chosen on dev F1. The final model was retrained on all 454 sentences for 9 epochs (709 s on CPU).

**Results (full test).** **P = 0.6785, R = 0.7334, F1 = 0.7049** – the best of the trained models: **+0.054 over BiLSTM-CRF and +0.123 over the CRF**. Pre-training on scientific text helps most where the BiLSTM had too little data: `chemicalelement` 0.8099 (BiLSTM 0.0588), `chemicalcompound` 0.7407, `protein` 0.6186, plus very strong `astronomicalobject` 0.9136, `scientist` 0.8731 and `academicjournal` 0.8538. The weak spots remain the rare or vague types: `book` 0.0 (3 mentions), `theory` 0.10, `event` 0.3929 and `misc` 0.4100.
"""

NOTES["E_NOTES"] = r"""### Part E – zero-shot vs. few-shot, parse failures, cost

**Set-up.** Model: **Qwen2.5-7B-Instruct** (official Q4_K_M GGUF), run locally with `llama-cpp-python` on CPU and greedy decoding (temperature 0). This is a real LLM, not a stand-in; no paid API was available. The sample has 100 test sentences (`random.Random(42)`), 551 gold entities. Both prompts contain a definition of each of the 18 types and require a JSON list `[{"text", "type"}]`. The **few-shot** prompt adds **5 training sentences** with their gold JSON as previous chat turns; they were chosen greedily to cover 16 of the 18 types. Each returned entity is mapped back to the first untagged token span whose text matches, ignoring case and spaces, and then scored with strict seqeval. All 200 prompts and raw responses are in `llm_ner_outputs.csv`.

**Results (100-sentence sample).**
- Zero-shot: P = 0.7613, R = 0.5499, **F1 = 0.6386**; **0 unparseable** responses; 4 items with an invalid type or text not found in the sentence.
- Few-shot: P = 0.7280, R = 0.6606, **F1 = 0.6927 (+0.0541)**; **1 unparseable** response; 14 invalid/unmatched items.

The examples mainly raise **recall**. They teach the dataset's annotation conventions, for instance which spans count as entities and that bare minor-planet names are `astronomicalobject` (0.72 → 0.84). `person` improves from 0.48 to 0.72 and `chemicalcompound` from 0.57 to 0.70. The LLM is excellent on types that its definitions describe well, such as `award` (0.95 zero-shot) and `academicjournal` (0.95 few-shot). It fails on the dataset-specific catch-all **`misc` (0.18 zero-shot, 0.10 few-shot)**, because a definition cannot convey what annotators put into "misc". The single parse failure and the invalid items show that JSON output from a 7B model is reliable but not perfect.

**Cost and latency.** The model needs no labelled training data beyond 5 examples. However, inference costs **8.48 s/sentence zero-shot and 11.02 s few-shot** on this CPU, because the few-shot prompt is 2.5× longer (983 vs 396 tokens on average). Fine-tuned SciBERT needs 40 ms/sentence, so the LLM is roughly 200–275× slower per sentence.
"""

NOTES["F_NOTES"] = r"""### Part F – which relations are hard, and how the methods differ

**Candidates and labels.** Every ordered pair of gold entities in a sentence is a candidate. Its label is the annotated relation (first label kept), with the 5 most frequent labels in train+validation (`role`, `physical`, `origin`, `part-of`, `related-to`) kept and the other 12 mapped to `OTHER`; un-annotated pairs are `NO_RELATION`. NO_RELATION is capped at 2× the positives with seed 42: train 14,821 → 3,390 and test 13,532 → 2,784. One test self-relation, where head and tail are the same span, cannot form a pair and is excluded.

**Method 1 – logistic regression.** Features: head and tail types and their type pair, direction, token distance (scaled and bucketed), number of entities in between, bag of words between the entities plus the first and last of them, the short between-phrase, and the head words of both entities. Class weights are balanced, and C = 10 was chosen by 5-fold CV. **Test macro-F1 = 0.4590.** The CV score (0.7411) is much higher than the test score because CV folds share sentences and documents, whose pairs are highly correlated.

**Method 2 – fine-tuned SciBERT with typed entity markers.** Input looks like `@ * scientist * August Kopff @ … # ^ scientist ^ Wolf # …`. The model is a 7-way classifier on `[CLS]`, trained with the same balanced class weights for 3 epochs (3,983 s on CPU). **Test macro-F1 = 0.5229 (+0.064)**; over the 6 relation classes alone (excluding NO_RELATION) it scores 0.4783 vs 0.3953.

**Hard relations.** `part-of` (F1 0.19 LogReg / 0.34 SciBERT) and `related-to` (0.21 / 0.29) are the hardest: both are semantically vague, are expressed with very different wording, and overlap with labels folded into `OTHER` (`type-of`, `named`, …). `OTHER` itself mixes 12 relation types, so it cannot have one consistent signature (0.46 / 0.44). `physical` is easiest (0.65 / 0.67) because its argument types (astronomical object or location) and cue words (*in*, *at*, *located*) are distinctive.

**Method differences.** SciBERT reads the whole sentence and, guided by the markers, can tell *who* did *what* to *whom*. This lifts `role` (0.40 → 0.55), `origin` (0.46 → 0.58) and `part-of` (0.19 → 0.34). It is, however, more trigger-happy: its NO_RELATION recall is only 0.7162, so about 28 % of true non-pairs receive a relation label, against 0.8797 for LogReg (NO_RELATION F1 0.79 vs 0.84). The confusion matrices above show where each model's mistakes fall.
"""

NOTES["G_NOTES"] = r"""### Error analysis (best model: fine-tuned SciBERT)

On the shared sample SciBERT makes **169 entity-level errors**: type confusion 73, boundary 36, spurious 32, missed (frequent type) 17, boundary + type 10, missed (rare type) 1. The full test split shows the same ranking (295 / 186 / 137 / 66 / 54 / 5). The 10 printed errors, grouped:

| group | errors | analysis |
|---|---|---|
| **Type confusion** | #1 `Viral TK` enzyme → misc; #7 `aciclovir` misc → chemicalcompound | The span is right but the label differs. #7 is arguably an annotation inconsistency: a drug labelled `misc` in gold, while the model's `chemicalcompound` is chemically correct. `misc` acts as a catch-all that overlaps every specific type, which is why type confusion is the largest category. |
| **Boundary errors** | #2 `cellular kinase` split into `cellular`[misc] + `kinase`[enzyme]; #8 `Robert 3D Del Naja` broken into person / misc / award | Multi-token names with a generic modifier or an unusual token (a digit inside a person name) break the B/I chain. |
| **Boundary + type** | #5 `Massive Attack` (organisation) predicted as `of Massive Attack` [misc] | This is an out-of-domain entity (a music band) in a "science" sentence, so the model has no domain knowledge to rely on. |
| **Spurious entities** | #3 `monophosphate` → chemicalcompound; #9 `viral` → misc | Both are scientific terms that are *plausible* entities but were not annotated, reflecting under-annotation of generic terms. |
| **Missed (frequent type)** | #4 `amplicon`, #10 `SNP` (both gold `misc`) | Generic technical terms are annotated as `misc` inconsistently, so the model learns to skip them. |
| **Missed rare type** | #6 `geophysics` (discipline, 42 training mentions) | Too few examples, and disciplines are lower-case common nouns with no surface cue. |

Overall, 8 of the 10 errors involve the vague `misc` class, as either the gold or the predicted label (#1, #2, #4, #5, #7, #8, #9, #10), and one more (#3) is a plausibly unannotated chemical. Most remaining errors therefore come from label definition and annotation consistency rather than from the model failing to read the sentence.

### Reflection – what the organisation should use

- **Accuracy.** Fine-tuned **SciBERT is the most accurate NER model: F1 0.7314 on the shared sample and 0.7049 on the full test**, against 0.6927 for the best LLM setting (few-shot), 0.6624 for BiLSTM-CRF and 0.6090 for the CRF. It also gives the best relation extractor (macro-F1 0.5229 vs 0.4590).
- **Cost.** SciBERT needs one fine-tuning run (709 s on a CPU) and then 40 ms/sentence. The few-shot LLM needs no training but takes **11.02 s/sentence**, about 275× slower. For a library of 1 million sentences that is roughly 11 CPU-hours for SciBERT vs. roughly 127 CPU-days for the LLM (1e6 × 0.040 s vs 1e6 × 11.02 s). The CRF is the cheapest option (13 s training, 1.2 ms/sentence) and remains a reasonable fallback at F1 0.6090.
- **Data size.** We have only 454 training sentences. The CRF learning curve is still rising (0.4643 → 0.5494 → 0.5766 → 0.5821 for 113 / 227 / 340 / 454 sentences), so more annotation would still help. Rare types are where all supervised models fail (`book`: 6 training mentions, F1 0 for every model). The LLM, which only needs a type definition, does well on some of them, e.g. `country` 0.933 few-shot vs 0.667 for SciBERT.
- **Domain.** General resources fit scientific entities less well: GloVe covers 97.1 % of all test tokens but only 93.2 % of entity tokens. SciBERT's vocabulary does *not* split entity words into fewer pieces than general BERT (1.792 vs 1.775 sub-tokens per entity word). Its advantage therefore comes from **pre-training on scientific text**, which gives contextual knowledge, rather than from tokenisation. The corpus also contains off-domain entities (bands, sports events) that hurt every model.
- **Recommendation.** Deploy **fine-tuned SciBERT** for NER and for relation classification in the research index. Use the **few-shot LLM offline as a pre-annotator**: zero-shot already reaches precision 0.7613, and suggestions corrected by curators grow the training set, especially for rare types and a clearer `misc` guideline. Re-fine-tune SciBERT periodically. Relation outputs (macro-F1 0.52) should be treated as suggestions for human review, not ground truth.
"""

NOTES["REFS"] = r"""- **Dataset:** CrossRE – Bassignana & Plank (2022), *CrossRE: A Cross-Domain Dataset for Relation Extraction*, Findings of EMNLP; HuggingFace `DFKI-SLT/cross_re` (config `science`, parquet revision).
- **Models:** SciBERT – Beltagy, Lo & Cohan (2019), `allenai/scibert_scivocab_cased`; `bert-base-cased` tokenizer (only for the vocabulary comparison); GloVe – Pennington, Socher & Manning (2014), 6B-100d via the `fse/glove-wiki-gigaword-100` mirror; **Qwen2.5-7B-Instruct** (Qwen Team, 2024), official `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M weights.
- **Methods:** BiLSTM-CRF – Lample et al. (2016) / Ma & Hovy (2016, char-CNN); typed entity markers for RE – Zhou & Chen (2021), *An Improved Baseline for Sentence-level Relation Extraction*.
- **Libraries:** HuggingFace `datasets` and `transformers`, PyTorch, `pytorch-crf`, `sklearn-crfsuite`, scikit-learn, `seqeval`, NLTK (averaged-perceptron POS tagger), gensim, `llama-cpp-python` (llama.cpp), pandas, matplotlib, joblib.
- **Code:** the notebook template and the `spans_to_bio`, `tokenize_and_align` and `llm_json_to_bio` helpers were provided by the course; `llm_json_to_bio` was extended (whitespace/case-insensitive matching, type normalisation, unmatched-item counting).
- **Prompts:** the zero-shot and few-shot prompts are defined in Part E and saved verbatim, with every raw response, in `llm_ner_outputs.csv`.
- **AI tools:** Claude (Anthropic) was used as a coding and writing assistant. All results come from the executed cells of this notebook, and every number quoted in the markdown was taken from the printed outputs.
"""

nb = json.load(open(NB, encoding="utf-8"))
done = set()
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    for k, v in NOTES.items():
        tag = "{{" + k + "}}"
        if tag in src:
            src = src.replace(tag, v.strip())
            done.add(k)
    c["source"] = src.splitlines(keepends=True)
missing = set(NOTES) - done
assert not missing, missing
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("filled:", sorted(done))
