import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.inference import build_feature_vector

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Music Virality Detector", page_icon="🎵", layout="wide")

# ─────────────────────────────────────────────
# LOAD MODEL (cached so it doesn't reload every rerun)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/trained/detection_model.pkl")
    threshold_info = joblib.load("models/trained/detection_threshold.pkl")
    return model, threshold_info

model, threshold_info = load_model()
optimal_threshold = threshold_info["threshold"]

st.title("🎵 Music Virality Detector")
st.write("Analyze if a music video is **currently viral** based on engagement signals")
st.info("💡 Tip: Get these stats from any YouTube video page")

# ─────────────────────────────────────────────
# SIDEBAR — Sample videos (defined before inputs so buttons can set state)
# ─────────────────────────────────────────────
st.sidebar.header("🎬 Try Sample Videos")

def _set_sample_video(values):
    for key, value in values.items():
        st.session_state[key] = value

st.sidebar.button(
    "💥 Viral Hit Example",
    on_click=_set_sample_video,
    kwargs={"values": {
        "views": 5_000_000, "likes": 250_000, "comments": 15_000,
        "subscribers": 500_000, "days_old": 7, "duration_seconds": 180,
    }},
)

st.sidebar.button(
    "📉 Struggling Video Example",
    on_click=_set_sample_video,
    kwargs={"values": {
        "views": 50_000, "likes": 1_000, "comments": 50,
        "subscribers": 5_000, "days_old": 14, "duration_seconds": 200,
    }},
)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
st.subheader("📊 Video Statistics")

col1, col2 = st.columns(2)
with col1:
    views = st.number_input("👁️ View Count", min_value=0, value=500000, step=10000, key="views")
    likes = st.number_input("👍 Like Count", min_value=0, value=25000, step=1000, key="likes")
    comments = st.number_input("💬 Comment Count", min_value=0, value=500, step=10, key="comments")
with col2:
    subscribers = st.number_input("👥 Subscriber Count", min_value=0, value=1000, step=100, key="subscribers")
    days_old = st.number_input("📅 Days Old", min_value=1, value=7, step=1, key="days_old")
    duration_seconds = st.number_input("🕒 Duration (seconds)", min_value=10, value=60, key="duration_seconds")

# ─────────────────────────────────────────────
# DERIVED FEATURES — built by the shared inference module so the columns always
# match what the model was trained on (19 features). Hardcoding them here is what
# previously broke inference (extra views_per_day / days_log columns).
# ─────────────────────────────────────────────
features = build_feature_vector(
    views=views, likes=likes, comments=comments,
    subscribers=subscribers, days_old=days_old, duration_seconds=duration_seconds,
)

# Interpretability values for the report below
like_rate = likes / views if views > 0 else 0.0
comment_rate = comments / views if views > 0 else 0.0
engagement_rate = (likes + 2 * comments) / views if views > 0 else 0.0
views_per_day = views / days_old
likes_per_day = likes / days_old
comments_per_day = comments / days_old
views_to_subs_ratio = views / subscribers if subscribers > 0 else 0.0
if subscribers < 10_000:
    channel_tier = 0
elif subscribers < 100_000:
    channel_tier = 1
elif subscribers < 1_000_000:
    channel_tier = 2
elif subscribers < 10_000_000:
    channel_tier = 3
else:
    channel_tier = 4

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
if st.button("Detect Virality", type="primary"):
    proba = model.predict_proba(features)[0, 1]
    is_viral = proba >= optimal_threshold

    # Stash so the expander/report below can reference the latest result after rerun
    st.session_state["last_proba"] = proba
    st.session_state["last_is_viral"] = is_viral

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Viral probability", f"{proba:.1%}")
    with m2:
        st.metric("Prediction", "✅ VIRAL" if is_viral else "❌ Not Viral")

    if proba >= 0.7:
        st.success("High confidence — this video has strong viral signals!")
    elif proba >= 0.4:
        st.warning("Moderate — could go either way")
    else:
        st.info("Low confidence — this video likely lacks viral signals")
else:
    st.caption("👆 Fill in the stats above and click **Detect Virality** to see results.")

# ─────────────────────────────────────────────
# EXPANDER — only meaningful after a prediction has been made
# ─────────────────────────────────────────────
if "last_proba" in st.session_state:
    with st.expander("🔍 See What the Model Saw"):
        st.write("**Engagement signals:**")
        st.write(f"- Like Rate: {like_rate*100:.2f}%")
        st.write(f"- Comment Rate: {comment_rate*100:.2f}%")
        st.write(f"- Engagement Rate: {engagement_rate*100:.2f}%")

        st.write("**Momentum Signals:**")
        st.write(f"- Views Per Day: {views_per_day:,.0f}")
        st.write(f"- Likes Per Day: {likes_per_day:,.0f}")
        st.write(f"- Comments Per Day: {comments_per_day:,.0f}")
        st.caption("No prior snapshot was provided, so growth/acceleration features default to 0.")

        st.write("**Channel Context:**")
        tier_labels = ['Nano (<10K)', 'Micro (10K-100K)', 'Mid (100K-1M)', 'Macro (1M-10M)', 'Mega (10M+)']
        st.write(f"- Channel Tier: {tier_labels[channel_tier]}")
        st.write(f"- Views-to-Subs Ratio: {views_to_subs_ratio:.2f}x")

        st.write("**Full feature vector sent to the model:**")
        st.dataframe(features.T.rename(columns={0: "value"}))

# ─────────────────────────────────────────────
# COMPARISON TABLE (always visible)
# ─────────────────────────────────────────────
st.subheader("📈 How Does This Compare?")
benchmark = pd.DataFrame({
    "Metric": ["Views/day", "Like Rate", "Engagement Rate", "Channel Size"],
    "This video": [
        f"{views_per_day:,.0f}",
        f"{like_rate*100:.1f}%",
        f"{engagement_rate*100:.1f}%",
        f"{subscribers:,}",
    ],
    "Viral Typical": ["50k+ /day", "4-8%", "3-6%", "100k+ subs"],
})
st.table(benchmark)

# ─────────────────────────────────────────────
# SIDEBAR — About
# ─────────────────────────────────────────────
st.sidebar.header("ℹ️ About This Model")
st.sidebar.write(f"""
**Detection Model**
- Algorithm: {threshold_info.get('model_name', 'XGBoost')}
- Features: 19 engineered signals
- Threshold: {optimal_threshold:.4f}

**Virality Definition:**
Video with 10M+ views
""")

st.sidebar.write("Built by Shayan Bhattacharjee")
st.sidebar.write("[GitHub](https://github.com/bhattacharjeeshayan860-netizen)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/shayan-bhattacharjee-860-netizen)")