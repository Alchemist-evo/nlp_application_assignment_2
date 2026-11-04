"""Build the submission notebook from nb_src.py (percent format).
usage: python build_nb.py OUT.ipynb [--debug] [--notes notes.json]"""
import json, re, sys
import nbformat

src_path = __file__.replace("build_nb.py", "nb_src.py")
out = sys.argv[1]
debug = "--debug" in sys.argv
notes = {}
if "--notes" in sys.argv:
    notes = json.load(open(sys.argv[sys.argv.index("--notes") + 1], encoding="utf-8"))

text = open(src_path, encoding="utf-8").read()
if debug:
    for k, v in {"CRF_MAX_ITER = 200": "CRF_MAX_ITER = 20", "LSTM_EPOCHS = 40": "LSTM_EPOCHS = 2",
                 "BERT_EPOCHS = 12": "BERT_EPOCHS = 1", "RE_EPOCHS = 3": "RE_EPOCHS = 1",
                 "SAMPLE_SIZE_CFG = 100": "SAMPLE_SIZE_CFG = 8", "LLM_MAX_NEW_TOKENS = 320": "LLM_MAX_NEW_TOKENS = 60"}.items():
        assert k in text, k
        text = text.replace(k, v)
    text = text.replace("grid = [(0.05, 0.01), (0.1, 0.05), (0.2, 0.01), (0.1, 0.1), (0.3, 0.05)]", "grid = [(0.05, 0.01), (0.1, 0.05)]")
    # keep debug RE / BERT runs short
    text = text.replace("re_train, n_neg_train = build_pairs(train_data)",
                        "re_train, n_neg_train = build_pairs(train_data[:60])\ntrain_data_re = train_data")
    text = text.replace("fit_data, BERT_EPOCHS, dev_set=dev_data", "fit_data[:40], BERT_EPOCHS, dev_set=dev_data")
    text = text.replace("bert_trainer = train_bert(train_data, BERT_BEST_EPOCH)", "bert_trainer = train_bert(train_data[:40], BERT_BEST_EPOCH)")

chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
cells = []
for i in range(1, len(chunks), 2):
    kind, body = chunks[i].strip(), chunks[i + 1].strip("\n")
    if kind == "[markdown]":
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        for k, v in notes.items():
            md = md.replace("{{" + k + "}}", v)
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbformat.write(nb, out)
print(f"wrote {out}: {len(cells)} cells")
