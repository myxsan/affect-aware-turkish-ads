"""Shared helpers for the Streamlit demo.

Self-contained: loads the fine-tuned BERTurk emotion classifier from
`outputs/berturk_emotion/` and the pre-built users/products/personas from
`outputs/`. All constants and the LLM ad-generation pipeline are inlined
here, so the demo has no external project-internal imports — it only depends
on the artefacts in `outputs/` produced by `colab_notebook.ipynb`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re as _re
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "berturk_emotion"
BASE_MODEL = "dbmdz/bert-base-turkish-cased"

# ─────────────────────────────── emotion class constants ───────────────────────────────
# (Inlined from the original data_emotion.py loader.)
EMOTION_LABELS = ["angry", "fear", "happy", "surprise", "sad"]
EMOTION_LABEL2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
EMOTION_ID2LABEL = {i: l for l, i in EMOTION_LABEL2ID.items()}

EMO_TR = {
    "angry":    "öfkeli",
    "fear":     "korkulu",
    "happy":    "mutlu",
    "surprise": "şaşkın",
    "sad":      "üzgün",
}
EMO_EMOJI = {
    "angry":    "😠",
    "fear":     "😨",
    "happy":    "😊",
    "surprise": "😲",
    "sad":      "😢",
}
EMO_COLOR = {
    "angry":    "#e45756",
    "fear":     "#b279a2",
    "happy":    "#54a24b",
    "surprise": "#eeca3b",
    "sad":      "#4c78a8",
}


# ─────────────────────────────── device ───────────────────────────────

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────── model ───────────────────────────────

@st.cache_resource(show_spinner="Loading BERTurk emotion classifier…")
def load_model():
    """Load tokenizer + fine-tuned BERTurk. Cached for the session.

    Tokenizer is loaded from a local `tokenizer/` subdirectory if present,
    otherwise from the HuggingFace Hub (the BERTurk base tokenizer is
    identical to ours since we didn't change the vocabulary during fine-tune).
    """
    device = pick_device()
    weights_path = MODEL_DIR / "pytorch_model.bin"
    if not weights_path.exists():
        st.error(
            f"❌ Fine-tuned BERTurk checkpoint not found at `{weights_path}`.\n\n"
            "Run `colab_notebook.ipynb` (§4) end-to-end in Colab and download "
            "`outputs/berturk_emotion/pytorch_model.bin` into the project root."
        )
        st.stop()
    local_tok_dir = MODEL_DIR / "tokenizer"
    tok_src = str(local_tok_dir) if local_tok_dir.exists() else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(EMOTION_LABELS),
        problem_type="single_label_classification",
    )
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return tok, model, device


@torch.no_grad()
def classify_texts(tok, model, device, texts, max_length: int = 64) -> np.ndarray:
    """Return [N, 5] softmax probabilities."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, len(EMOTION_LABELS)))
    enc = tok(
        list(texts),
        truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits.detach().float().cpu()
    return torch.softmax(logits, dim=-1).numpy()


# ─────────────────────────────── data loaders ───────────────────────────────

def _missing_file_warning(path: Path) -> list:
    st.warning(
        f"`{path.name}` not found in `outputs/`. "
        f"Re-run the relevant section of `colab_notebook.ipynb` and "
        f"download the updated `outputs/` folder."
    )
    return []


@st.cache_data(show_spinner=False)
def load_users():
    path = OUTPUTS_DIR / "users.json"
    if not path.exists():
        return _missing_file_warning(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_products():
    path = OUTPUTS_DIR / "products.json"
    if not path.exists():
        return _missing_file_warning(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_personas():
    path = OUTPUTS_DIR / "personas.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────── persona renderer ───────────────────────────────

def render_persona_full(emotion_dist: dict, keywords: list, examples: list,
                        dominant_conf: float, low_conf: float = 0.45) -> str:
    """Render a full Turkish persona block in the same format as the notebook §8."""
    sorted_dist = sorted(emotion_dist.items(), key=lambda kv: -kv[1])
    dom = sorted_dist[0][0]
    pct_line = ", ".join(f"%{round(100*v):d} {EMO_TR[k]}" for k, v in sorted_dist)
    if dominant_conf >= low_conf:
        affect = (
            f"Bu Türkçe Twitter kullanıcısı son paylaşımlarında ağırlıklı "
            f"olarak **{EMO_TR[dom]}** bir duygu durumu sergilemektedir."
        )
    else:
        affect = (
            "Bu Türkçe Twitter kullanıcısında belirgin bir duygu eğilimi "
            "gözlenmemekte; farklı duygular arasında dengeli bir dağılım vardır."
        )
    dist_line = "Duygu dağılımı: " + pct_line + "."
    kw_line = ("Sıkça geçen içerik terimleri: " + ", ".join(keywords) + "."
               if keywords else "")
    ex_line = "Örnek paylaşımları:\n" + "\n".join(f'  - "{t}"' for t in examples)
    return "\n\n".join(p for p in [affect, dist_line, kw_line, ex_line] if p)


# ─────────────────────────────── LLM ad generator ───────────────────────────────
# Inlined from the notebook §10 (Groq backend + deterministic stub fallback).

SYSTEM_PROMPT = (
    "Sen kısa ve etkili Türkçe reklam metinleri yazan bir kreatif yazarsın. "
    "Kuralların:\n"
    "1. Reklam 40-70 kelime arasında ve TEK paragraf olmalı.\n"
    "2. Kullanıcının ruh halini ya da kişiliğini AÇIKÇA söyleme — buna göre "
    "uygun bir TON seç ama 'sen şu duygudasın' gibi ifadeler kullanma.\n"
    "3. Türkçe akıcı ve doğal olsun, çeviri kokmasın.\n"
    "4. Ürünün adını ve en az bir özelliğini metne yedir.\n"
    "5. Sonuç sadece reklam metnidir; başlık, açıklama veya emoji ekleme."
)
_GROQ_MIN_INTERVAL = 2.5  # ~30 RPM free-tier safety margin
_CACHE_DIR = OUTPUTS_DIR / "ads_cache"


def _cache_key(system: str, user: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode()); h.update(b"\n---\n")
    h.update(system.encode()); h.update(b"\n---\n"); h.update(user.encode())
    return h.hexdigest()[:24]


def _build_user_prompt(persona_text: str, product: dict) -> str:
    feats = ", ".join(product.get("key_features", []))
    return ("Aşağıdaki kullanıcı profili ve ürüne uygun bir reklam metni yaz.\n\n"
            f"KULLANICI PROFİLİ:\n{persona_text}\n\n"
            f"ÜRÜN:\nİsim: {product['name']}\n"
            f"Açıklama: {product['description']}\n"
            f"Öne çıkan özellikler: {feats}\n"
            f"Fiyat: {product.get('price_hint','—')}\n\n"
            "REKLAM METNİ:")


def _gen_stub(system: str, user: str, model: str) -> str:
    hints = {"öfkeli": "kontrolü geri al", "korkulu": "huzurlu bir nefes",
             "üzgün": "kendine bir mola ver", "şaşkın": "yeni bir keşif",
             "mutlu": "iyi anı sürdür"}
    earliest = (10**9, "doğal")
    for emo, hint in hints.items():
        idx = user.find(emo)
        if idx != -1 and idx < earliest[0]:
            earliest = (idx, hint)
    tone = earliest[1]
    name, feature = "ürün", ""
    for line in user.splitlines():
        if line.startswith("İsim:"):
            name = line.split(":", 1)[1].strip()
        if line.startswith("Öne çıkan"):
            feature = line.split(":", 1)[1].split(",")[0].strip()
    return (f"{tone.capitalize()} için {name}. "
            f"{feature.capitalize()} ile gününe küçük bir fark kat. "
            f"Hemen dene, kendine iyi gel.")


def _gen_groq(system: str, user: str, model: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    last = getattr(_gen_groq, "_last_call", 0.0)
    elapsed = time.time() - last
    if elapsed < _GROQ_MIN_INTERVAL:
        time.sleep(_GROQ_MIN_INTERVAL - elapsed)

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.7, max_tokens=600,
        )
    try:
        resp = _call()
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate" in msg or "quota" in msg:
            time.sleep(30)
            resp = _call()
        else:
            raise
    _gen_groq._last_call = time.time()
    return resp.choices[0].message.content.strip()


_BACKENDS = {"stub": _gen_stub, "groq": _gen_groq}
_DEFAULT_MODELS = {"stub": "stub-v1", "groq": "llama-3.3-70b-versatile"}


def generate_ad(persona_text: str, product: dict,
                backend: str = "stub", model: str | None = None,
                use_cache: bool = True) -> tuple:
    """Return (ad_text, meta_dict) with disk cache keyed by SHA-256(system+user+backend:model)."""
    if model is None:
        model = _DEFAULT_MODELS[backend]
    system = SYSTEM_PROMPT
    user = _build_user_prompt(persona_text, product)
    key = _cache_key(system, user, f"{backend}:{model}")
    cache_file = _CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        with open(cache_file, encoding="utf-8") as fh:
            hit = json.load(fh)
        return hit["ad_text"], {**hit["meta"], "cache_hit": True}

    t0 = time.time()
    ad_text = _BACKENDS[backend](system, user, model)
    meta = {"backend": backend, "model": model, "cache_hit": False,
            "latency_seconds": round(time.time() - t0, 3), "key": key}
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump({"ad_text": ad_text, "meta": meta,
                   "system": system, "user": user}, fh,
                  ensure_ascii=False, indent=2)
    return ad_text, meta


@st.cache_resource(show_spinner=False)
def llm_backend_info():
    """Detect available backend, return (backend_name, model_name, status_msg)."""
    if os.environ.get("GROQ_API_KEY"):
        return "groq", "llama-3.3-70b-versatile", "✓ Groq (Llama-3.3-70B) ready"
    return "stub", "stub-v1", "⚠ No GROQ_API_KEY — using deterministic stub"


def generate_ad_safe(persona_text: str, product: dict,
                     backend: str, model: str | None = None):
    """Friendly wrapper around generate_ad — catches network/quota errors."""
    try:
        return generate_ad(persona_text, product, backend=backend, model=model)
    except Exception as exc:
        return f"⚠ Generation failed: {exc}", {
            "backend": backend, "model": model or "default",
            "cache_hit": False, "latency_seconds": 0.0, "key": "",
            "error": str(exc),
        }


# ─────────────────────────────── Layer-2 ad metrics (compact) ───────────────────────────────

_WORD_RE = _re.compile(r"[a-zçğıöşüâîû]+", flags=_re.IGNORECASE)
_EMOJI_RE = _re.compile(
    "[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F]", flags=_re.UNICODE,
)
_CALLOUT_RE = _re.compile(
    r"\b(siz|sen)\s+\w*(öfkeli|sinirli|üzgün|mutlu|korkulu|şaşkın|kaygılı)|"
    r"\b(öfkeli|sinirli|üzgün|mutlu|korkulu|şaşkın|kaygılı)\s+"
    r"(görünüyorsun|hissediyorsun|olmalısın|olduğun)",
    flags=_re.IGNORECASE,
)
EMOTION_AD_LEXICON = {
    "angry":    {"öfke", "sinir", "kontrol", "rahatla", "rahatlat", "boşalt",
                 "patla", "stres", "gerginlik", "sakin", "tepki", "soğukkanlı"},
    "fear":     {"korku", "huzur", "güven", "güvenli", "sakin", "endişe",
                 "koru", "korur", "nefes", "rahat", "emin", "garanti"},
    "happy":    {"mutlu", "neşe", "keyif", "kutla", "harika", "eğlence",
                 "sürdür", "iyi", "güzel", "parla", "ışılda", "an", "anı"},
    "surprise": {"sürpriz", "şaşkın", "keşfet", "yeni", "merak", "heyecan",
                 "beklenmedik", "fark", "büyülen"},
    "sad":      {"moral", "mola", "kendine", "destek", "umut", "yalnız",
                 "şefkat", "sıcak", "ısı", "rahatla", "huzur", "iyileş", "yenile"},
}


def quick_metrics(ad: str, product: dict, dominant_emotion: str | None) -> dict:
    """Compact subset of the notebook §12 Layer-2 metrics."""
    toks = [t.lower() for t in _WORD_RE.findall(ad)]
    n = len(toks)
    tok_set = set(toks)
    feature_lists = product.get("key_features", [])
    name_ok = product["name"].lower() in ad.lower()
    feat_ok = any(any(w in tok_set for w in _WORD_RE.findall(f.lower()))
                  for f in feature_lists)
    callout = bool(_CALLOUT_RE.search(ad))
    emoji = bool(_EMOJI_RE.search(ad))
    emo_hits = 0
    if dominant_emotion in EMOTION_AD_LEXICON:
        lex = EMOTION_AD_LEXICON[dominant_emotion]
        long_stems  = {w for w in lex if len(w) >= 4}
        short_exact = {w for w in lex if len(w) < 4}
        for t in toks:
            if t in short_exact:
                emo_hits += 1
            elif any(t.startswith(s) for s in long_stems):
                emo_hits += 1
    return {
        "word_count": n,
        "in_band":    40 <= n <= 70,
        "product_named": name_ok,
        "feature_named": feat_ok,
        "explicit_callout": callout,
        "has_emoji": emoji,
        "emotion_density": round(emo_hits / max(n, 1), 4),
    }
