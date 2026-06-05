# Affect-Aware Personalized Advertisement Generation from Turkish Tweets

**Author:** Musa Yüksel — 21050911018

**Course:** Neural Networks — Final Project

**Department:** Software Engineering · Ankara Yıldırım Beyazıt University

End-to-end Turkish ad-personalization pipeline with a falsifiable 3-arm
ablation that isolates the contribution of the affective signal from prompt
length and prompt presence.

---

## Headline results

| layer | result |
|---|---|
| Emotion classifier — **BERTurk fine-tuned**, 5-class | **macro F1 = 0.9949** |
| Δ vs strongest classical baseline (TF-IDF + Linear SVM) | +0.0050 |
| Δ vs cross-lingual XLM-R-base | +0.0025 (within noise) |
| Frozen-encoder ablation (no fine-tune) | 0.7802 — *worse than TF-IDF*; fine-tuning is doing the work |
| Personalization (LLM-as-judge, 1–5) — full vs generic | **4.8 vs 2.2** (+2.6, tight std) |
| Personalization driven by affect content vs prompt length | **73 % content / 27 % length** |
| creep_factor — full vs generic (lower is better) | 2.0 vs 3.8 — affect-aware ads are **less** creepy than the controls |

Full methodology, numbers, per-rubric checklist and the LLM-as-judge prompt
breakdown are in
[`SENG428_PROJECT_REPORT_revised.docx`](SENG428_PROJECT_REPORT_revised.docx).

---

## Repository layout

```
.
├── README.md                              ← this file
├── requirements.txt                       ← pip dependencies (used by notebook + demo)
├── .gitignore
│
├── SENG428_PROJECT_REPORT_revised.docx    ← final report (the submission deliverable)
│
├── colab_notebook.ipynb                   ← self-contained Colab notebook (42 cells, 100 KB)
│
├── outputs/                               ← all generated artefacts (~426 MB, BERTurk dominates)
│   ├── berturk_emotion/
│   │   ├── pytorch_model.bin              ← 442 MB (the trained classifier the demo loads)
│   │   ├── tokenizer/
│   │   ├── confusion_matrix.png, history.png, test_metrics.json
│   ├── xlmr_emotion/                      ← test_metrics + confusion matrix (weights deleted to save 1 GB)
│   ├── baselines/                         ← TF-IDF + LogReg + Linear SVM results
│   ├── ads_cache/                         ← SHA-256-keyed cache of all Groq calls (so re-runs are free)
│   ├── users.json, user_vectors.json, personas.json, products.json
│   ├── ads_results.{json,csv}, eval_results.json, judge_results.json, ...
│   ├── model_comparison{,_extended}.{csv,png}
│   ├── sweep_results.csv, sweep_heatmap.png
│   ├── robustness_results.csv, robustness_curve.png
│   └── eda_emotion.png
│
└── demo/                                  ← Streamlit demo app
    ├── app.py                             ← 3-tab interactive demo
    ├── _utils.py                          ← model loading, LLM ad-gen, Layer-2 metrics (self-contained)
    ├── requirements.txt
    ├── .streamlit/config.toml
    └── README.md                          ← demo-specific run guide
```

The notebook is the canonical source of every experimental result; the
`outputs/` folder is its byproduct. The demo only reads from `outputs/`
and doesn't depend on any other project subfolder.

---

## How to run

There are two ways to run the project. **The notebook is the most complete;
the Streamlit demo is the most fun to present.**

### Option A — Google Colab (full pipeline, recommended)

This runs every section end-to-end (classifier training, baselines, XLM-R,
hyperparameter sweep, frozen-encoder ablation, robustness, 30-ad sweep,
LLM-as-judge). The notebook auto-mounts Google Drive so trained checkpoints
survive runtime restarts.

1. Open [`colab_notebook.ipynb`](colab_notebook.ipynb) in Google Colab.
2. **Runtime → Change runtime type → GPU (T4)**.
3. *(Recommended)* Get a free Groq API key at
   [console.groq.com/keys](https://console.groq.com/keys) (no credit card).
   In Colab, click the **🔑 Secrets** icon in the left sidebar, add a new
   secret named `GROQ_API_KEY`, paste your key, and toggle **Notebook
   access** ON. Without a key the notebook still runs end-to-end using a
   deterministic stub.
4. **Runtime → Run all**. The first cell will pop a Drive-mount prompt —
   click Allow. Total wall-clock on a T4: ~15–20 min (BERTurk + XLM-R
   training dominate; the Groq calls run in 2–3 min thanks to a 2.5 s/call
   rate-limit pacing).
5. Outputs land in `/content/drive/MyDrive/outputs/`. They persist
   across runtime restarts, so re-runs are fast (everything caches). If
   you want to run the local Streamlit demo against the same artefacts,
   download that folder and rename it `outputs/` at the project root.

### Option B — Streamlit demo (fastest, most visual)

Three interactive tabs:

1. **🔍 Classify a tweet** — paste any Turkish tweet, see BERTurk's softmax over the 5 emotion classes.
2. **👤 Inspect a user** — pick one of the 25 pre-built synthetic users, see per-tweet predictions, the aggregated affect vector, and all three persona variants (full / random-donor / generic).
3. **📝 Generate personalized ads** — pick a user × product, generate three ads side-by-side with Layer-2 compliance + personalization metrics under each.

```bash
# from project root
python -m venv .venv && source .venv/bin/activate     # macOS / Linux
# or:  py -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

export GROQ_API_KEY=gsk_...                            # optional; falls back to stub if missing

streamlit run demo/app.py
```

Opens at <http://localhost:8501>. **First model load is the slow part:
~10–20 s on Apple Silicon MPS, ~5 s on a Colab T4 / CUDA.** After that
everything is session-cached.

The demo loads the BERTurk checkpoint from `outputs/berturk_emotion/`. If
you haven't trained yet, run Option A first and download `outputs/` from
your Colab Drive.

Auto-detected device priority: CUDA → Apple Silicon MPS → CPU.

**Note for Apple Silicon users:** there is a known PyTorch MPS `int64`
round-trip corruption bug that misreports macro-F1 as ~0.20 during training.
The notebook ships with the workaround pre-applied (see `evaluate()` in §4);
the fix is a no-op on CUDA. Story is in the report's "Engineering
observations" subsection.

---

## LLM backends

The ad generator (notebook §10 and `demo/_utils.py`) supports one real-LLM
backend plus a deterministic stub fallback:

| backend | model | cost | rate limit | when to use |
|---|---|---|---|---|
| **groq** | Llama-3.3-70B-Versatile | **free, no credit card** | 30 RPM | default; what the report numbers come from |
| stub | deterministic template | none | none | grader-friendly fallback when no key is set |

Generated ads are cached on disk by `SHA-256(system + user + backend:model)`
in `outputs/ads_cache/`, so re-runs are free and the experiment is fully
reproducible.

---

## Reproducibility

- All `random.seed`, `np.random.seed`, `torch.manual_seed` set to **42**.
- 80/10/10 stratified split with `random_state=42` reused across **all
  classifiers** (TF-IDF + LogReg, TF-IDF + Linear SVM, BERTurk full
  fine-tune, frozen-encoder BERTurk + LogReg, XLM-R fine-tune) so the
  comparison is exact on the same held-out test set.
- Reported headline numbers are from a CUDA T4 run; on Apple Silicon MPS,
  per-user aggregator results may flip a borderline user by ≤5 %
  confidence due to nondeterministic MPS reductions. The full discussion
  is in the report's limitations section.

---

## Rubric coverage

| § | requirement | location |
|---|---|---|
| 3.1 | Problem definition | report §1 |
| 3.2 | Dataset description | report §3.1 |
| 3.3 | Baseline method | TF-IDF + LogReg + Linear SVM (report §3.2, §4.1; notebook §3) |
| 3.4 | NN/Transformer/LLM-based method | BERTurk + XLM-R + Llama-3.3 (report §3.3–§3.8) |
| 3.5 | Training / adaptation strategy | report §3.3 + notebook §4 |
| 3.6 | ≥2 metrics + confusion matrix | accuracy + macro F1 + per-class F1 + CM (report §4) |
| 3.7 | Short literature review | report §2 |
| 3.8 | Limitations and ethical considerations | report §5 |
| 3.9 | Individual contribution statement | report §6 |
| §4 (rec) | Multi-Transformer comparison | report §4.1 (BERTurk vs XLM-R) |
| §4 (rec) | Hyperparameter sweep | report §4.1 (2 × 2 LR × batch grid) |
| §4 (rec) | Ablation study | report §4.2–§4.3 (persona ablation) + §4.1 (frozen encoder) |
| §4 (rec) | Robustness | report §4.1 (character-level noise injection) |
| §4 (rec) | Prompt vs fine-tuned comparison | implicit (BERTurk fine-tuned + Llama prompted) |
| §5 (agentic clause) | Experimental comparison vs non-agentic baseline | persona ablation (§4.2–§4.3): full vs random vs generic |
