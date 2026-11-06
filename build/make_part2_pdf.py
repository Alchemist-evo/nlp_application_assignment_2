"""Generate Assignment2_SituatedLearning_Group_86.pdf (Part 2, 5 marks).
Evidence numbers are read from the EXECUTED notebook so they match it exactly.
usage: python build/make_part2_pdf.py"""
import json, re, html
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

NB = "Assignment2_Programming_Group_86.ipynb"
OUT = "Assignment2_SituatedLearning_Group_86.pdf"

# ---------- pull evidence from the executed notebook ----------
nb = json.load(open(NB, encoding="utf-8"))
def outputs_text():
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        for o in c.get("outputs", []):
            if o["output_type"] == "stream": yield "".join(o["text"])
            elif "data" in o and "text/plain" in o["data"]: yield "".join(o["data"]["text/plain"])
ALL = "\n".join(outputs_text())

def grab(pattern, cast=float):
    m = re.search(pattern, ALL)
    if not m: raise SystemExit(f"pattern not found: {pattern}")
    return cast(m.group(1))

# comparison table rows: "<idx>  <Model>  P  R  F1  F1full ..."
cmp = {}
for name in ["CRF", "BiLSTM-CRF", r"Transformer \(SciBERT\)", "LLM zero-shot", "LLM few-shot"]:
    m = re.search(rf"\d\s+{name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)", ALL)
    cmp[name.replace("\\", "")] = m.groups()
lr_macro = grab(r"Method 1 – Logistic regression, TEST macro-F1 = ([\d.]+)")
bre_macro = grab(r"Method 2 – fine-tuned SciBERT \(entity markers\), TEST macro-F1 = ([\d.]+)")
n_train_sent = grab(r"Training set size: (\d+) sentences", int)
n_train_ment = grab(r"sentences / (\d+) entity mentions", int)

llm = pd.read_csv("llm_ner_outputs.csv")
ex = llm[(llm.setting == "few-shot (5)") & llm.parsed_ok & (llm.n_entities_returned.between(3, 6))].iloc[0]

# ---------- document ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1f3b6f"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=12.6, spaceAfter=3)
SM = ParagraphStyle("SM", parent=B, fontSize=8.4, leading=10.5)
MONO = ParagraphStyle("MONO", parent=B, fontName="Courier", fontSize=7.6, leading=9.4)

def P(t, s=B): return Paragraph(t, s)

def field_table(rows, w=(4.2 * cm, 12.8 * cm)):
    t = Table([[P(f"<b>{k}</b>", SM), P(v, SM)] for k, v in rows], colWidths=w)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f8"))]))
    return t

def arch_diagram():
    steps = ["Article PDFs /\nabstracts", "Text & sentence\nextraction", "NER\n(SciBERT)", "Entity linking\n(PubChem, ORCID)",
             "Relation\nclassifier", "Knowledge graph\n+ search index"]
    w, h, gap = 2.45 * cm, 1.35 * cm, 0.42 * cm
    d = Drawing(len(steps) * (w + gap), h + 0.9 * cm)
    for i, s in enumerate(steps):
        x = i * (w + gap)
        d.add(Rect(x, 0.5 * cm, w, h, rx=4, ry=4, fillColor=colors.HexColor("#e3ecfa"), strokeColor=colors.HexColor("#1f3b6f")))
        for j, line in enumerate(s.split("\n")):
            d.add(String(x + w / 2, 0.5 * cm + h - 0.5 * cm - j * 0.38 * cm, line, fontSize=7.2, textAnchor="middle"))
        if i < len(steps) - 1:
            y = 0.5 * cm + h / 2
            d.add(Line(x + w, y, x + w + gap, y, strokeColor=colors.black))
            d.add(Polygon([x + w + gap, y, x + w + gap - 4, y + 3, x + w + gap - 4, y - 3], fillColor=colors.black))
    d.add(String(0, 0.05 * cm, "Human-in-the-loop: curators review low-confidence entities/relations; corrections are fed back as training data.",
                 fontSize=7.2))
    return d

story = []
story += [P("Assignment 2 – Part 2: Situated Learning", H1),
          P("AIML ZG519 – NLP Applications &nbsp;|&nbsp; Selected topic: <b>Information Extraction</b> "
            "&nbsp;|&nbsp; Group 86", B)]

MEMBERS = [  # (name, BITS ID, contribution)
    ("Prajwal Shetty K P", "2025AE05434", "Part A data prep &amp; EDA; Part D SciBERT fine-tuning; notebook integration"),
    ("Neha Yadav", "2025AE05463", "Part B CRF; Part C BiLSTM-CRF"),
    ("Poornima Prakash Patil", "2025AE05411", "Part E LLM NER; Part F relation extraction"),
    ("Narumalla Pavan Kumar", "2025AE05457", "Part G comparison &amp; error analysis; Part 2 Situated Learning report"),
]
story += [P("Student details", H2), field_table([
    ("Group number", "86"),
    ("Team member names", "<br/>".join(f"{i}. {n} ({b})" for i, (n, b, _) in enumerate(MEMBERS, 1))),
    ("BITS email id", "<br/>".join(f"{b.lower()}@wilp.bits-pilani.ac.in" for _, b, _ in MEMBERS)),
    ("Task allocation", "<br/>".join(f"<b>{n}:</b> {c}" for n, _, c in MEMBERS)),
    ("Industry / Domain", "Scientific publishing and R&amp;D (chemicals / materials / life-science research)"),
    ("Organisation type", "R&amp;D knowledge-management team of a science-based manufacturer or publisher"),
])]
story += [P("<i>Status statement (Section 6 of the brief):</i> we distinguish <b>observed practice</b> (information extraction "
            "that is publicly documented and in production in scientific publishing, cited below) from our <b>proposed "
            "application</b> for an R&amp;D knowledge-management team. The proposal is evidenced by the working prototype "
            "built in Part 1 of this assignment; no confidential organisational data is used – all examples come from the "
            "public CrossRE dataset.", SM)]

story += [P("5.1 Industry Context – Anchor", H2), field_table([
    ("Industry / Organisation", "Scientific publishing and corporate R&amp;D. Publishers and public literature services already "
     "run large-scale entity extraction (e.g. NCBI <b>PubTator</b> annotates genes, chemicals, diseases, species and variants "
     "across PubMed; <b>Europe PMC</b> exposes text-mined annotations on articles; the open-source <b>ChemDataExtractor</b> "
     "toolkit extracts chemical entities and properties from papers). Inside a manufacturer, the R&amp;D knowledge-management "
     "team curates the internal and licensed article library."),
    ("Selected NLP concept", "Information Extraction: Named Entity Recognition (NER) + Relation Extraction (RE)"),
    ("Application / use case", "Automatically index every article by the compounds, elements, proteins/enzymes, scientists, "
     "theories, awards, institutions and journals it mentions, plus the relations between them (role, origin, part-of, "
     "physical, related-to), to power a searchable research index / knowledge graph."),
    ("Problem addressed", "Researchers cannot quickly answer questions such as <i>which compounds were studied together with "
     "enzyme X</i> or <i>which scientists and institutions worked on theory Y</i>. Keyword search misses synonyms and does not "
     "know entity types; manual tagging by librarians is slow, inconsistent and does not scale to thousands of new papers a month."),
    ("Users / stakeholders", "Bench scientists and project leads (search &amp; discovery); KM team / librarians (curation); "
     "IP &amp; patent analysts (prior-art, freedom-to-operate); regulatory &amp; product-safety teams (where a substance is "
     "discussed); R&amp;D management (expertise location, collaboration mapping); IT / data-platform team (operations)."),
])]

story += [P("5.2 Concept in Practice – Apply", H2), field_table([
    ("Data / input", "Full-text articles and abstracts (PDF/XML) from licensed journals and internal technical reports. "
     "Training/evaluation data: an annotated sentence set; our prototype uses CrossRE-science "
     f"({n_train_sent} training sentences, {n_train_ment} entity mentions, 18 entity types, 17 relation types)."),
    ("NLP approach / concept", "Sequence labelling NER with BIO tags (entity-level strict F1 evaluation); relation "
     "classification over candidate entity pairs; LLM prompting as a zero-/few-shot alternative and for pre-annotation."),
    ("Tool / model / framework", "HuggingFace Transformers (SciBERT <font face='Courier'>allenai/scibert_scivocab_cased</font>), "
     "sklearn-crfsuite (CRF baseline), PyTorch BiLSTM-CRF with GloVe, Qwen2.5-7B-Instruct via llama.cpp (local LLM), "
     "seqeval / scikit-learn for evaluation. Production additions: a PDF-to-text parser (e.g. GROBID), "
     "an annotation tool (e.g. Label Studio), a graph store / search index (e.g. Neo4j, OpenSearch)."),
    ("Workflow / architecture", "See diagram: ingest &rarr; text &amp; sentence extraction &rarr; NER &rarr; entity "
     "normalisation/linking to identifiers &rarr; relation classification for co-occurring entity pairs &rarr; "
     "triples loaded into a knowledge graph and a faceted search index &rarr; researcher UI. Low-confidence outputs are "
     "routed to curators, whose corrections become new training data."),
    ("Output / result", "Per-article JSON records of typed entities and relation triples, e.g. "
     "(<i>scientist</i>, role, <i>university</i>) or (<i>chemicalcompound</i>, part-of, <i>chemicalcompound</i>), "
     "exposed as search facets and graph queries."),
    ("Evidence", "Prototype from Part 1 (executed notebook, numbers below are copied from its printed outputs) and the "
     "saved LLM prompt/response log <font face='Courier'>llm_ner_outputs.csv</font>."),
])]
story += [Spacer(1, 4), arch_diagram()]

rows = [["Model (Part 1 prototype)", "P", "R", "F1 (100-sent. sample)", "F1 (full test)", "Inference ms/sent."]]
for k, (p, r, f, ff, tt, inf, pf) in cmp.items():
    rows.append([k, p, r, f, ff if ff != "NaN" else "n/a *", f"{float(inf):.1f}"])
t = Table(rows, colWidths=[4.6 * cm, 1.6 * cm, 1.6 * cm, 3.2 * cm, 2.6 * cm, 3.2 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b6f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
story += [Spacer(1, 4), P("<b>Evidence 1 – NER results (strict entity-level, seqeval) on CrossRE-science test data:</b>", SM), t,
          P("* The LLM was evaluated only on the shared 100-sentence test sample (about 8–11 s per sentence on CPU "
            "makes the full 400-sentence test impractical), so it has no full-test F1. Compare all five models on the "
            "100-sentence sample column.", SM),
          P(f"<b>Evidence 2 – relation extraction (gold entities, 7 classes):</b> logistic-regression macro-F1 = {lr_macro:.3f}; "
            f"fine-tuned SciBERT with typed entity markers macro-F1 = {bre_macro:.3f}.", SM),
          P("<b>Evidence 3 – representative input/output (few-shot LLM, raw response saved in the CSV):</b>", SM),
          P("Input: " + html.escape(ex.sentence), MONO), P("Output: " + html.escape(ex.raw_response[:600]), MONO)]

f1s = {k: float(v[2]) for k, v in cmp.items()}
best = max(f1s, key=f1s.get)
story += [P("5.3 Industry Analysis – Analyse", H2), field_table([
    ("Key benefits / value", "(1) <b>Findability:</b> type-aware facets (compound, enzyme, scientist, award) replace "
     "keyword guessing. (2) <b>Scale &amp; consistency:</b> a fine-tuned model tags every new article in milliseconds "
     f"(measured {float(cmp[best][5]):.0f} ms/sentence on CPU for the best model) with the same guidelines every time. "
     "(3) <b>Discovery:</b> relation triples reveal co-occurrence and provenance networks (who worked where, which compound "
     "is part of which material). (4) <b>Re-use:</b> the same index feeds patent analytics, regulatory look-ups and RAG "
     "assistants."),
    ("Limitations / problems", f"(1) <b>Accuracy ceiling with little data:</b> with only {n_train_sent} annotated sentences the best "
     f"model reached F1 = {f1s[best]:.3f} on the shared sample, so outputs need human review before being trusted for "
     "IP or safety decisions. (2) <b>Long-tail types</b> (theory, book, chemical element) have very few examples and are "
     "the most error-prone. (3) <b>Ambiguity:</b> scientist vs. person, organisation vs. university, misc catch-all. "
     f"(4) <b>Relations are harder than entities</b> (best macro-F1 {max(lr_macro, bre_macro):.3f}). (5) Prompted LLMs are "
     "cheap to start but slow on CPU, sometimes return unparseable JSON and can invent entity types. "
     "(6) PDF parsing errors (tables, formulae, ligatures) propagate downstream."),
    ("Evidence of effectiveness / impact", f"In the prototype (F1 on the shared sample: {', '.join(f'{k} {v:.3f}' for k, v in f1s.items())}) "
     f"the fine-tuned SciBERT was the most accurate model at {float(cmp[best][5]):.0f} ms/sentence, while the few-shot LLM came "
     f"second without any training data, beating the CRF and BiLSTM-CRF but at {float(cmp['LLM few-shot'][5]) / 1000:.1f} s/sentence. "
     "This supports a fine-tuned model in production, with the LLM used for bootstrapping annotations. "
     "Publicly, services such as PubTator and Europe PMC demonstrate that automated entity annotation is viable at "
     "literature scale and is used by curators and researchers for search and database curation."),
])]

story += [P("5.4 Reflection and Future Scope – Reflect", H2), field_table([
    ("Major challenge / gap", "<b>Annotation scarcity for the organisation's own entity types and relations.</b> Generic "
     "scientific NER covers common types, but each company cares about its own materials, product codes and internal "
     "projects; there is little labelled data, and rare types/relations score lowest. Entities are also not yet linked "
     "to canonical identifiers, so synonyms (e.g. 'ATP' vs 'Adenosine triphosphate') remain separate nodes."),
    ("Recommended improvement", "An <b>active-learning annotation loop</b>: use the local LLM (few-shot) to pre-annotate new "
     "articles, have curators correct them in an annotation tool, and periodically re-fine-tune SciBERT on the growing set; "
     "add an <b>entity-linking</b> step to PubChem/ChEBI/ORCID/ROR identifiers. This converts cheap LLM suggestions into "
     "high-quality training data while keeping the fast, accurate fine-tuned model in production."),
    ("Potential future use", "A knowledge-graph-grounded <b>research assistant (Graph-RAG)</b>: researchers ask natural-language "
     "questions ('which catalysts containing ruthenium were reported with yields above 90 %?') and get answers with "
     "citations to the extracted triples and source sentences; plus alerting when new papers mention a watched compound, "
     "competitor or scientist."),
])]

story += [P("Acknowledgements and sources", H2), P(
    "Dataset: CrossRE (Bassignana &amp; Plank, 2022; HF <font face='Courier'>DFKI-SLT/cross_re</font>). Models: SciBERT "
    "(Beltagy et al., 2019), GloVe (Pennington et al., 2014), Qwen2.5-7B-Instruct (Qwen team, 2024). Public systems "
    "referenced: NCBI PubTator (Wei et al.; PubTator3, 2024), Europe PMC text-mined annotations, ChemDataExtractor "
    "(Swain &amp; Cole, 2016). Tools: HuggingFace Transformers, sklearn-crfsuite, llama.cpp, reportlab. "
    "AI assistance: Claude (Anthropic) was used to help draft code and text; all numbers are taken from our executed notebook.", SM)]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="Assignment 2 – Situated Learning (Information Extraction)")
doc.build(story)
print("wrote", OUT)
