# Streamlit Demo — Affect-Aware Turkish Ad Generator

Interactive demo of the Neural Networks final project: a 4-stage pipeline
(BERTurk emotion classifier → per-user aggregator → Turkish persona →
Llama-3.3-70B ad generator) with the 3-arm ablation visible side-by-side.

---

## What's in here

Three tabs:

1. **🔍 Classify a tweet** — type any Turkish tweet, get the BERTurk softmax
   distribution across the 5 emotion classes (angry / fear / happy / surprise / sad).
2. **👤 Inspect a user** — pick one of the 25 pre-built synthetic users; see
   per-tweet predictions, the aggregated affect vector, and all three persona
   variants (full / random-donor / generic).
3. **📝 Generate personalized ads** — pick a user × product, generate three ads
   side-by-side with Layer-2 compliance + personalization metrics under each.

## Prerequisites

You need the **trained BERTurk checkpoint** at
`../outputs/berturk_emotion/pytorch_model.bin` (~442 MB) plus the
pre-built `users.json`, `products.json`, `personas.json`, and
`user_vectors.json` in `../outputs/`. All of these are produced by running
`colab_notebook.ipynb` end-to-end in Colab and downloading the resulting
`/content/drive/MyDrive/outputs/` folder into the project root as
`outputs/`.

If you only need to verify the pipeline shape without the trained model,
the first two tabs require the BERTurk weights but tab 3 (ad generation)
will work with the deterministic stub fallback as long as the user/product
JSONs exist.

## Run locally

```bash
# from project root
pip install -r requirements.txt

# (optional but recommended) enable the real LLM backend
export GROQ_API_KEY=gsk_...       # get one free at console.groq.com/keys

streamlit run demo/app.py
```

Opens at <http://localhost:8501> by default.

Without `GROQ_API_KEY`, the ad-generator tab falls back to a deterministic
stub backend that's defined inside `_utils.py` (no network call, ads are
templated). The first two tabs (classifier + user inspector) work fully
offline and don't need any API key.

## Performance notes

- First model load is the slow part: ~10–20 s on Apple Silicon MPS,
  ~5 s on a Colab T4. After that everything is cached for the session
  via `st.cache_resource`.
- Each ad generation hits Groq at ~3 s/call, paced to stay under the
  free-tier 30 RPM. Repeated calls with the same persona+product+model
  combo are served from the SHA-256-keyed disk cache in
  `../outputs/ads_cache/` and are instant.

## File layout

```
demo/
├── app.py              ← main Streamlit app (3 tabs)
├── _utils.py           ← model loading, persona renderer, LLM ad-gen, Layer-2 metrics
├── requirements.txt    ← pinned versions (same as project-root requirements.txt)
├── .streamlit/
│   └── config.toml     ← theme + server config
└── README.md           ← this file
```

`_utils.py` is **fully self-contained** — it doesn't import from any other
project folder. All emotion-class constants and the LLM ad-generation
pipeline are inlined locally; the demo only depends on the JSON/PNG/model
artefacts in `../outputs/`.
