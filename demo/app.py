"""Streamlit demo for the Affect-Aware Turkish Ad-Generation pipeline.

Three tabs:
  1. Tweet Classifier  — type/paste one tweet → BERTurk emotion + probability bars.
  2. User Persona      — pick one of the 25 synthetic users → see their bundle,
                          aggregated affect vector, and three persona variants.
  3. Ad Generator      — pick a user × product → generate three side-by-side ads
                          (full / random / generic) via Groq Llama or stub, with
                          Layer-2 metrics under each.

Run from project root:
    streamlit run demo/app.py
"""
from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from _utils import (
    EMOTION_LABELS, EMOTION_LABEL2ID, EMO_TR, EMO_EMOJI, EMO_COLOR,
    classify_texts, generate_ad_safe, llm_backend_info,
    load_model, load_personas, load_products, load_users,
    pick_device, quick_metrics, render_persona_full,
)

st.set_page_config(
    page_title="Turkish Affect-Aware Ads — Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── sidebar ───────────────────────────────

with st.sidebar:
    st.title("🎯 Affect-Aware Ads")
    st.caption("Turkish tweet → affect vector → personalized ad. "
               "Final project — Neural Networks, AYBÜ.")

    st.divider()
    st.subheader("Pipeline status")

    device = pick_device()
    st.markdown(f"**Device:** `{device}`")

    backend, model_name, status_msg = llm_backend_info()
    st.markdown(f"**LLM backend:** `{backend}` ({model_name})")
    if backend == "stub":
        st.warning("No GROQ/GEMINI key found. Ads will be deterministic stubs.\n\n"
                   "To enable real Groq Llama-3.3-70B:\n"
                   "```bash\nexport GROQ_API_KEY=gsk_...\nstreamlit run demo/app.py\n```")
    else:
        st.success(status_msg)

    st.divider()
    st.subheader("Project")
    st.markdown(
        "- 📓 [`colab_notebook.ipynb`](../colab_notebook.ipynb) — full notebook\n"
        "- 📄 [`SENG428_PROJECT_REPORT_revised.docx`](../SENG428_PROJECT_REPORT_revised.docx) — final report\n"
        "- 📊 [`README.md`](../README.md) — project README"
    )

    st.divider()
    st.caption("Musa Yüksel · 21050911018 · Software Engineering")


# ─────────────────────────────── header ───────────────────────────────

st.title("🎯 Affect-Aware Turkish Ad Generator")
st.markdown(
    "Live demo of a 4-stage pipeline: **emotion classifier → user vector → "
    "Turkish persona → LLM ad generator**. The classifier is a fine-tuned "
    "BERTurk reaching **0.9949 macro-F1**; the ad generator uses Groq's free "
    "Llama-3.3-70B by default."
)

tab_classify, tab_user, tab_ads = st.tabs(
    ["🔍 1 — Classify a tweet",
     "👤 2 — Inspect a user",
     "📝 3 — Generate personalized ads"]
)


# ╭──────────────────────────── TAB 1 ────────────────────────────╮
# │ Tweet classifier                                              │
# ╰───────────────────────────────────────────────────────────────╯

with tab_classify:
    st.subheader("Run BERTurk on a single Turkish tweet")

    examples = {
        "— pick an example —": "",
        "happy (mutluyum)": "bugün harika hissediyorum mutluyum",
        "angry (sinirliyim)": "yine aynı saçmalık delireceğim sinirliyim",
        "fear (korkuyorum)": "gece evde tek başımayım korkuyorum",
        "surprise (şaşırdım)": "böyle bir sürpriz beklemiyordum şaşkına döndüm",
        "sad (üzgünüm)": "sevdiklerimi özledim bugün moralim çok bozuk",
    }
    col_ex, col_clear = st.columns([4, 1])
    with col_ex:
        ex_choice = st.selectbox("Quick examples", list(examples.keys()), key="ex")
    with col_clear:
        st.markdown("&nbsp;")
        clear = st.button("Clear")

    default_text = "" if clear else examples[ex_choice]
    tweet = st.text_area("Tweet text", value=default_text, height=100,
                         placeholder="Türkçe bir tweet yazın veya yapıştırın…")

    if st.button("🔍 Classify", type="primary", disabled=not tweet.strip()):
        tok, model, device = load_model()
        probs = classify_texts(tok, model, device, tweet.strip())[0]
        pred_idx = int(np.argmax(probs))
        pred = EMOTION_LABELS[pred_idx]
        conf = float(probs[pred_idx])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {EMO_EMOJI[pred]} **{pred}** / *{EMO_TR[pred]}*")
            st.metric("Confidence", f"{conf:.1%}")
        with c2:
            df_probs = pd.DataFrame({
                "emotion (EN)": EMOTION_LABELS,
                "emotion (TR)": [EMO_TR[e] for e in EMOTION_LABELS],
                "probability": probs,
            }).sort_values("probability", ascending=False)
            st.dataframe(
                df_probs,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "probability": st.column_config.ProgressColumn(
                        "probability", format="%.3f", min_value=0.0, max_value=1.0,
                    ),
                },
            )

        with st.expander("How does this work?"):
            st.markdown(
                "BERTurk (`dbmdz/bert-base-turkish-cased`, 110M params) fine-tuned "
                "for 3 epochs on **3,121 Turkish tweets** with AdamW + linear "
                "warmup-and-decay. The displayed probabilities are the **softmax "
                "over the [CLS] classification head**, with `max_length=64` and "
                "no preprocessing beyond tokenization."
            )


# ╭──────────────────────────── TAB 2 ────────────────────────────╮
# │ Inspect a synthetic user                                      │
# ╰───────────────────────────────────────────────────────────────╯

with tab_user:
    st.subheader("Aggregate a 12-tweet bundle into a per-user affect vector")
    users = load_users()
    if not users:
        st.stop()

    # build labels for the selectbox
    def _user_label(u):
        if u["type"] == "balanced":
            return f"{u['user_id']}  ·  balanced (control)"
        return (f"{u['user_id']}  ·  skewed {EMO_EMOJI[u['dominant_emotion']]} "
                f"{u['dominant_emotion']} ({EMO_TR[u['dominant_emotion']]})")

    uid_to_user = {u["user_id"]: u for u in users}
    pick = st.selectbox("Pick a synthetic user",
                        [_user_label(u) for u in users], index=5)  # u05 = first skewed
    selected_uid = pick.split("·")[0].strip()
    u = uid_to_user[selected_uid]

    tok, model, device = load_model()
    texts = [t["text"] for t in u["tweets"]]
    probs = classify_texts(tok, model, device, texts)
    argmax = probs.argmax(axis=1)
    mean_probs = probs.mean(axis=0)
    pred_dom = EMOTION_LABELS[int(mean_probs.argmax())]
    dom_conf = float(mean_probs.max())

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Ground-truth dominant (by construction)")
        if u["dominant_emotion"]:
            st.markdown(f"### {EMO_EMOJI[u['dominant_emotion']]} "
                        f"{u['dominant_emotion']} ({EMO_TR[u['dominant_emotion']]})")
        else:
            st.markdown("### — *balanced (no dominant)*")
    with c2:
        st.markdown("##### Aggregator prediction")
        if dom_conf >= 0.45:
            st.markdown(f"### {EMO_EMOJI[pred_dom]} {pred_dom} "
                        f"({EMO_TR[pred_dom]})  ·  conf={dom_conf:.2f}")
        else:
            st.markdown(f"### ❔ no clear dominant  ·  conf={dom_conf:.2f} (< 0.45)")

    st.divider()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### Per-tweet predictions")
        df_tweets = pd.DataFrame({
            "text": texts,
            "true": [t["true_emotion"] for t in u["tweets"]],
            "pred": [EMOTION_LABELS[i] for i in argmax],
            "conf": [float(probs[i, argmax[i]]) for i in range(len(texts))],
        })
        df_tweets["✓"] = (df_tweets["true"] == df_tweets["pred"]).map(
            {True: "✅", False: "❌"})
        st.dataframe(
            df_tweets[["✓", "text", "true", "pred", "conf"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "conf": st.column_config.ProgressColumn(
                    "conf", format="%.2f", min_value=0.0, max_value=1.0),
                "text": st.column_config.TextColumn("text", width="large"),
            },
        )
        n_correct = (df_tweets["true"] == df_tweets["pred"]).sum()
        st.caption(f"Per-tweet accuracy on this bundle: **{n_correct}/{len(texts)}**")

    with c2:
        st.markdown("##### Aggregated emotion distribution")
        df_dist = pd.DataFrame({
            "emotion": [f"{EMO_EMOJI[e]} {e}" for e in EMOTION_LABELS],
            "mean prob": mean_probs,
        }).sort_values("mean prob", ascending=False)
        st.dataframe(
            df_dist, hide_index=True, use_container_width=True,
            column_config={
                "mean prob": st.column_config.ProgressColumn(
                    "mean prob", format="%.3f", min_value=0.0, max_value=1.0),
            },
        )

    st.divider()
    st.markdown("##### Three persona variants (used by the ablation)")

    # build all three personas live from the current vector
    keywords_top6 = (
        Counter(w.lower() for t in texts for w in t.split()
                if len(w) > 4).most_common(6)
    )
    keywords = [w for w, _ in keywords_top6]
    # examples: 2 from dominant + 1 other (fall back to random if balanced)
    rng = random.Random(42 + hash(u["user_id"]) % 1000)
    if dom_conf >= 0.45:
        dom_idx_local = [i for i, t in enumerate(u["tweets"])
                         if EMOTION_LABELS[argmax[i]] == pred_dom]
        oth_idx_local = [i for i, t in enumerate(u["tweets"])
                         if EMOTION_LABELS[argmax[i]] != pred_dom]
        dom_idx_local.sort(key=lambda i: -float(probs[i, argmax[i]]))
        chosen = dom_idx_local[:2] + (
            [rng.choice(oth_idx_local)] if oth_idx_local else dom_idx_local[2:3]
        )
    else:
        chosen = rng.sample(range(len(texts)), 3)
    examples = [texts[i] for i in chosen[:3]]
    full_persona = render_persona_full(
        {EMOTION_LABELS[i]: float(mean_probs[i]) for i in range(len(EMOTION_LABELS))},
        keywords, examples, dom_conf,
    )

    # for the random variant, pick a different user as donor
    other_users = [v for v in users
                   if v["user_id"] != u["user_id"]
                   and (v["dominant_emotion"] != u.get("dominant_emotion"))]
    donor = rng.choice(other_users) if other_users else users[0]
    donor_texts = [t["text"] for t in donor["tweets"]]
    donor_probs = classify_texts(tok, model, device, donor_texts)
    donor_mean = donor_probs.mean(axis=0)
    donor_argmax = donor_probs.argmax(axis=1)
    donor_keywords = [
        w for w, _ in Counter(
            x.lower() for t in donor_texts for x in t.split() if len(x) > 4
        ).most_common(6)
    ]
    donor_examples = donor_texts[:3]
    random_persona = render_persona_full(
        {EMOTION_LABELS[i]: float(donor_mean[i]) for i in range(len(EMOTION_LABELS))},
        donor_keywords, donor_examples, float(donor_mean.max()),
    )

    generic_persona = (
        "Bu Türkçe konuşan bir sosyal medya kullanıcısıdır. "
        "Kişisel tercih, duygu durumu veya özel ilgi alanları hakkında "
        "ek bilgi bulunmamaktadır."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**🟢 full** — real affect")
        st.markdown(full_persona)
    with pc2:
        st.markdown(f"**🟡 random** — donor `{donor['user_id']}` "
                    f"(*{donor.get('dominant_emotion','—')}*)")
        st.markdown(random_persona)
    with pc3:
        st.markdown("**⚪ generic** — no affect info")
        st.markdown(generic_persona)

    # stash for tab 3
    st.session_state.setdefault("personas_for_user", {})
    st.session_state["personas_for_user"][u["user_id"]] = {
        "full": full_persona, "random": random_persona, "generic": generic_persona,
        "dominant": pred_dom if dom_conf >= 0.45 else None,
        "donor_id": donor["user_id"],
    }


# ╭──────────────────────────── TAB 3 ────────────────────────────╮
# │ Generate ads (full / random / generic side-by-side)           │
# ╰───────────────────────────────────────────────────────────────╯

with tab_ads:
    st.subheader("Generate three ads side-by-side — full vs random vs generic")
    users = load_users()
    products = load_products()
    personas_disk = load_personas()  # the pre-built ones from outputs/personas.json
    if not users or not products:
        st.stop()

    # we prefer disk personas (consistent with notebook); fall back to live-built
    personas_disk_by_uid = ({p["user_id"]: p for p in personas_disk}
                            if personas_disk else {})

    c1, c2 = st.columns(2)
    with c1:
        def _user_label(u):
            if u["type"] == "balanced":
                return f"{u['user_id']}  ·  balanced"
            return (f"{u['user_id']}  ·  {EMO_EMOJI[u['dominant_emotion']]} "
                    f"{u['dominant_emotion']}")
        pick_u = st.selectbox(
            "Choose a user", [_user_label(u) for u in users], index=5,
            help="Skewed users have a known dominant emotion; balanced are controls.",
        )
        chosen_uid = pick_u.split("·")[0].strip()
    with c2:
        def _prod_label(p):
            return f"{p['product_id']}  ·  {p['name']}  ({p['category_en']})"
        pick_p = st.selectbox(
            "Choose a product", [_prod_label(p) for p in products], index=1,
            help="10 fictional Turkish products spanning diverse categories.",
        )
        chosen_pid = pick_p.split("·")[0].strip()

    user = next(u for u in users if u["user_id"] == chosen_uid)
    product = next(p for p in products if p["product_id"] == chosen_pid)

    # backend status
    backend, model_name, _ = llm_backend_info()
    st.caption(f"LLM backend: **{backend}** ({model_name}) — first call may take a few seconds "
               "due to rate-limit pacing; subsequent identical calls hit the disk cache.")

    if st.button("📝 Generate 3 ads", type="primary"):
        # use disk personas if available; otherwise re-render live
        if chosen_uid in personas_disk_by_uid:
            personas = personas_disk_by_uid[chosen_uid]["personas"]
            dominant = personas_disk_by_uid[chosen_uid].get("true_dominant")
        elif chosen_uid in st.session_state.get("personas_for_user", {}):
            stored = st.session_state["personas_for_user"][chosen_uid]
            personas = {k: stored[k] for k in ("full", "random", "generic")}
            dominant = stored["dominant"]
        else:
            st.warning("Visit tab 2 first to build this user's personas, "
                       "OR re-run §8 of `colab_notebook.ipynb` to produce personas.json.")
            st.stop()

        ads = {}
        with st.spinner(f"Calling {backend}…"):
            for variant in ("full", "random", "generic"):
                ad, meta = generate_ad_safe(personas[variant], product,
                                            backend=backend, model=model_name)
                ads[variant] = (ad, meta)

        cols = st.columns(3)
        labels = {"full": "🟢 full", "random": "🟡 random", "generic": "⚪ generic"}
        for col, variant in zip(cols, ("full", "random", "generic")):
            ad, meta = ads[variant]
            with col:
                st.markdown(f"#### {labels[variant]}")
                st.write(ad)

                m = quick_metrics(ad, product, dominant)
                tag_cache = "💾 cached" if meta.get("cache_hit") else "⚡ live"
                st.caption(f"{tag_cache}  ·  latency {meta.get('latency_seconds', 0)}s")

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("words", m["word_count"],
                              delta="in band" if m["in_band"] else "off band",
                              delta_color="normal" if m["in_band"] else "off")
                with mc2:
                    st.metric("emo density", f"{m['emotion_density']:.3f}")
                with mc3:
                    callout = "⚠ yes" if m["explicit_callout"] else "✓ none"
                    st.metric("explicit callout", callout,
                              delta_color="inverse" if m["explicit_callout"]
                              else "normal")

                with st.expander("compliance details"):
                    st.markdown(
                        f"- product name mentioned: {'✅' if m['product_named'] else '❌'}\n"
                        f"- ≥1 feature mentioned: {'✅' if m['feature_named'] else '❌'}\n"
                        f"- has emoji (forbidden): {'⚠ yes' if m['has_emoji'] else '✓ none'}\n"
                        f"- explicit emotion callout: "
                        f"{'⚠ yes' if m['explicit_callout'] else '✓ none'}"
                    )

        st.divider()
        st.markdown("##### What to look for in this comparison")
        st.markdown(
            "- **full** should reference the user's emotional register through "
            "*tonal* word choice, without explicitly naming the emotion.\n"
            "- **random** should sound coherent but for a *different* user — "
            "tone won't match the user's actual emotion.\n"
            "- **generic** should be category-generic copy with no affective grounding.\n\n"
            "In our Layer-3 evaluation (10 cells, judge = Llama-3.3-70B), "
            "personalization scored **4.8 (full) > 2.9 (random) > 2.2 (generic)** "
            "on a 5-point scale — i.e. 73% of the lift comes from the *content* "
            "of the persona, only 27% from the prompt being structurally larger."
        )
