import pandas as pd, numpy as np, re, json, os, openai, warnings
from sentence_transformers import SentenceTransformer
import umap, hdbscan
from rapidfuzz import fuzz
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity 

# ---------- global resources ----------
EMB_MODEL    = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # all-MiniLM-L6-v2

OPENAI_MODEL = "gpt-4o-mini"
client       = openai.OpenAI()

# ---------- tiny helpers --------------
def _first_meaningful(df_sub):
    """Return first USER turn with >=3 words; else first."""
    for row in df_sub.itertuples():
        if len(str(row.text).split()) >= 3:
            return row.text
    return df_sub.iloc[0].text

def _cluster(sentences):
    emb = EMB_MODEL.encode(sentences, normalize_embeddings=True)
    red = umap.UMAP(n_components=10, metric="cosine").fit_transform(emb)
    return hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit(red).labels_

def _gpt_label(texts):
    """Return a 2-3 word noun phrase for one intent cluster."""
    prompt = ("Return a 2-3 word noun phrase that captures this user intent:\n"
              + "\n".join(f"- {t}" for t in texts[:40]) + "\nLabel:")
    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6, temperature=0
        ).choices[0].message.content.strip()
        return out.title()
    except Exception:
        return "Unknown"



# ---------- public function ----------
def discover_themes(chat_df: pd.DataFrame):
    """Upload CSV → return list of unique intent themes."""
    first_user = (chat_df[chat_df.speaker=="user"]
                  .sort_values(["conversation_id","turn_index"])
                  .groupby("conversation_id")
                  .apply(_first_meaningful))
    sentences  = first_user.tolist()

    labels     = _cluster(sentences)
    raw_map = {}
    for cid in sorted(set(labels) - {-1}):          # skip noise cluster -1
        cluster_sents = [sent for sent, lab in zip(sentences, labels) if lab == cid]
        raw_map[cid]  = _gpt_label(cluster_sents)

    # semantic merge via simple GPT call
    vals = sorted(set(raw_map.values()) - {"Unknown"})
    prompt = f"""
      You will receive a list of tentative intent labels.

      {json.dumps(vals, indent=2)}

      Some labels describe the **same intent**.
      Return ONLY a JSON array (no markdown) containing ONE canonical label
      for each distinct intent.  Example output:

      ["Flight Booking Request", "Flight Cancellation Request", "Hotel Add-On"]

      Rules:
      • Merge synonyms, plural/singular, etc.
      • Each label ≤ 3 words, Title Case, no pronouns/verbs.
      • No duplicates, no commentary, no markdown fences.
      """
    merged = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=120, temperature=0
    ).choices[0].message.content
    unique_themes = json.loads(re.sub(r"```.*?```", "", merged, flags=re.S))

    # final alphabetical clean-up
    unique_themes = sorted(set(t.title() for t in unique_themes))
    return unique_themes


# 6️⃣  GPT examples --------------------------------------------------
def examples_for_themes(themes, n=3):
    def _examples(intent):
        prompt = (f"Give {n} user sentences (≤15 words) that express the "
                  f'intent "{intent}". Return JSON array, no markdown.')
        txt = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=120, temperature=0.7
        ).choices[0].message.content
        clean = re.sub(r"```.*?```","", txt, flags=re.S).strip()
        try:
            return json.loads(clean)
        except Exception:
            # fallback: split lines
            return [l.strip() for l in clean.split("\n") if l.strip()][:n]

    return {t: [t]+_examples(t) for t in themes}

# 7️⃣  Bucket sentences ---------------------------------------------
def bucket_sentences(sentences, theme_variants):
    all_phr = [p for var in theme_variants.values() for p in var]
    vec_all = EMB_MODEL.encode(all_phr, normalize_embeddings=True)

    # map theme -> indices in vec_all
    idx=0; t2i={}
    for t,var in theme_variants.items():
        t2i[t]=list(range(idx, idx+len(var))); idx+=len(var)

    buckets = defaultdict(list)
    for s in sentences:
        v = EMB_MODEL.encode(s, normalize_embeddings=True)
        best_t, best = None, -1
        for t,ix in t2i.items():
            score = cosine_similarity(v.reshape(1,-1), vec_all[ix]).mean()
            if score > best: best, best_t = score, t
        buckets[best_t].append((s, best))
    # sort
    for t in buckets: buckets[t].sort(key=lambda x: x[1], reverse=True)
    return buckets

# 8️⃣  Conversation-level KPIs --------------------------------------
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
SENT_ANALYZER = SentimentIntensityAnalyzer()
import pandas as pd, re
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_distances

def conversation_kpis(df: pd.DataFrame, conv2theme: dict,
                      first_user_sentences: list, model=EMB_MODEL):
    """Return KPI summary and semantic_diversity per theme."""
    # a. per-conversation heuristics
    FALLBACK = re.compile(r"\bsorry\b|don.?t understand|not sure", re.I)
    THANKS   = re.compile(r"\bthanks|thank you|great|solved|works\b", re.I)

    stats = {}
    prev_user = {}

    for _, r in df.sort_values(["conversation_id","turn_index"]).iterrows():
        cid = r.conversation_id
        s   = stats.setdefault(cid, dict(fb=0,retry=0,success=False,
                                         sent_first=None,sent_last=None))
        if r.speaker=="bot" and FALLBACK.search(r.text): s["fb"]+=1
        if r.speaker=="user":
            # retry test
            if cid in prev_user and fuzz.token_set_ratio(prev_user[cid], r.text)>=80:
                s["retry"]+=1
            prev_user[cid]=r.text
            # sentiment
            score = SENT_ANALYZER.polarity_scores(r.text)["compound"]
            s["sent_first"]=s["sent_first"] or score
            s["sent_last"]=score
            if THANKS.search(r.text): s["success"]=True

    # conv length
    conv_len = df.groupby("conversation_id").turn_index.max()+1
    for cid,l in conv_len.items(): stats[cid]["len"]=l

    metrics = pd.DataFrame(stats).T
    metrics["theme"] = metrics.index.map(conv2theme)

    # semantic diversity (pair-wise cosine distance of first-turn vectors)
    div = {}
    # embed once
    vecs = model.encode(first_user_sentences, normalize_embeddings=True)
    for theme in set(conv2theme.values()):
        idxs = [i for i,(cid,th) in enumerate(conv2theme.items()) if th==theme]
        div[theme] = cosine_distances(vecs[idxs]).mean() if len(idxs)>1 else 0.0

    kpi = (metrics.groupby("theme")
           .agg(n=("len","count"),
                mean_len=("len","mean"),
                sd_len=("len","std"),
                success_pct=("success","mean"),
                avg_fb=("fb","mean"),
                avg_retry=("retry","mean"),
                sent_delta=("sent_last","mean"))
           .round(2)
           .assign(semantic_diversity=lambda d:d.index.map(div)))

    return kpi.reset_index().rename(columns={"index":"theme"})
