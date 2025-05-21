import streamlit as st, pandas as pd, plotly.express as px, os
from analyzer import discover_themes, examples_for_themes, bucket_sentences
import plotly.express as px, plotly.graph_objects as go
from analyzer import conversation_kpis        # new import

st.set_page_config("Intent Studio", layout="wide")
st.title("🤖 Chatbot Intent Studio")

# API key gate
if "OPENAI_API_KEY" not in os.environ:
    key = st.text_input("Enter OpenAI API key", type="password")
    if key: os.environ["OPENAI_API_KEY"] = key; st.experimental_rerun()

csv = st.file_uploader("Upload chat CSV", type="csv")

# persistent session state
state = st.session_state
for k in ("themes", "variants", "buckets", "pie",
          "conv2theme", "sentences", "kpi_df", "df"):
    st.session_state.setdefault(k, None)

# 1️⃣ discover themes
if csv and st.button("Discover Themes") or (csv and state.themes is None):
    df = pd.read_csv(csv)
    st.session_state.df = df
    with st.spinner("Detecting intents…"):
        state.themes = discover_themes(df)
    st.success("Themes ready! Edit below ⬇️")

# 2️⃣ editable multiselect
if state.themes:
    st.subheader("🎯 Intent themes")
    keep = st.multiselect("Keep / drop themes", options=state.themes,
                          default=state.themes)
    new = st.text_input("Add custom theme")
    if st.button("➕ Add") and new:
        keep.append(new.title())
    state.themes = sorted(set(keep))
    st.write(state.themes)

    # 3️⃣ generate examples + bucket when user clicks
    if st.button("Run bucketing & show chart"):
        with st.spinner("Generating examples & bucketing…"):
            state.variants = examples_for_themes(state.themes, n=3)

            # rebuild first-user sentences + conv_ids
            first_df = (state.df[state.df.speaker=="user"]
                        .sort_values(["conversation_id","turn_index"])
                        .groupby("conversation_id").first())
            first_sents = first_df.text.tolist()
            conv_ids    = first_df.index.tolist()

            # bucket sentences
            state.buckets = bucket_sentences(first_sents, state.variants)

            # map conv_id → theme (best theme for its first sentence)
            conv2theme = {}
            for theme, items in state.buckets.items():
                for sent,_ in items:
                    cid = conv_ids[first_sents.index(sent)]
                    conv2theme[cid] = theme

            # store for KPI stage
            state.conv2theme = conv2theme
            state.sentences  = first_sents


        # 4️⃣ pie chart
        sizes = {t: len(v) for t,v in state.buckets.items()}
        fig = px.pie(values=sizes.values(), names=sizes.keys(),
                     title="Intent Distribution")
        state.pie = fig

# display pie + buckets
if state.pie:
    st.plotly_chart(state.pie, use_container_width=True)

    with st.expander("See bucketed sentences"):
        for t, items in state.buckets.items():
            st.markdown(f"**{t} ({len(items)})**")
            st.write(pd.DataFrame(items, columns=["sentence","similarity"]))



# ── KPI + charts ────────────────────────────────────────────
if state.pie:
    # run KPI calc once and cache
    if state.kpi_df is None:
      with st.spinner("Computing KPIs…"):
          state.kpi_df = conversation_kpis(state.df,
                                          state.conv2theme,
                                         state.sentences)

    kpi_df = state.kpi_df

    st.subheader("Conversation KPIs")
    st.dataframe(kpi_df, use_container_width=True)

    # 1. bar w/ error bars
    fig_len = go.Figure()
    fig_len.add_bar(x=kpi_df.theme, y=kpi_df.mean_len,
                    error_y=dict(type='data', array=kpi_df.sd_len))
    fig_len.update_layout(title="Mean Conversation Length (turns)",
                          yaxis_title="Turns")
    st.plotly_chart(fig_len, use_container_width=True)

    # 2. success pct stacked bar
    fig_succ = px.bar(kpi_df, x="theme", y="success_pct",
                      title="Success Rate per Intent", labels={"success_pct":"Success %"})
    st.plotly_chart(fig_succ, use_container_width=True)

    # 3. scatter: sentiment Δ vs. diversity
    fig_sent = px.scatter(kpi_df, x="semantic_diversity", y="sent_delta",
                          text="theme", title="Sentiment Change vs. Diversity",
                          labels={"semantic_diversity":"Semantic Diversity",
                                  "sent_delta":"Avg Sentiment Δ"})
    fig_sent.update_traces(textposition="top center")
    st.plotly_chart(fig_sent, use_container_width=True)

