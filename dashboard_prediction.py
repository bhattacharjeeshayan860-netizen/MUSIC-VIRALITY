import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Music Virality Predictor", page_icon="🎵", layout="wide")

# ─────────────────────────────────────────────
# LOAD MODEL (cached so it doesn't reload every rerun)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/trained/prediction_model.pkl")
    threshold_info = joblib.load("models/trained/prediction_threshold.pkl")
    return model, threshold_info

model, threshold_info = load_model()
optimal_threshold = threshold_info["threshold"]

st.title("🎵 Music Virality Predictor")
st.write("Analyze if a music video has **viral potential** based on engagement signals")
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
    duration_seconds = st.number_input("🕒 Duration (seconds)", min_value=10, value=210, key="duration_seconds")

# ─────────────────────────────────────────────
# DERIVED FEATURES (computed every rerun — all 25, matching training exactly)
# ─────────────────────────────────────────────
like_rate = likes / views if views > 0 else 0.0
comment_rate = comments / views if views > 0 else 0.0
engagement_rate = (likes + 2 * comments) / views if views > 0 else 0.0
comment_like_ratio = comments / likes if likes > 0 else 0.0

views_per_day = views / days_old
likes_per_day = likes / days_old
comments_per_day = comments / days_old

# No second snapshot available in single-snapshot mode -> momentum features default to 0
# (the imputer was fit to handle this; it is NOT the same as leaving them out)
views_growth_rate = 0.0
views_acceleration = 0.0
views_diff = 0.0
likes_diff = 0.0
engagement_diff = 0.0

view_count_log = np.log1p(views)
like_count_log = np.log1p(likes)
comment_count_log = np.log1p(comments)
subscriber_count_log = np.log1p(subscribers)

views_to_subs_ratio = views / subscribers if subscribers > 0 else 0.0
views_per_day_per_sub = (views_per_day / subscribers) if subscribers > 0 else 0.0

days_log = np.log1p(days_old)
duration_log = np.log1p(duration_seconds)
is_short_video = 1.0 if duration_seconds < 180 else 0.0  # < 3 min, typical short-form cutoff

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

if days_old <= 7:
    age_bucket = 0
elif days_old <= 30:
    age_bucket = 1
elif days_old <= 90:
    age_bucket = 2
elif days_old <= 365:
    age_bucket = 3
else:
    age_bucket = 4

# Single-snapshot mode: rank 1 of 1
snapshot_rank = 1
snapshot_count = 1

# Exact 25 features, exact order the model was trained on
FEATURE_COLUMNS = [
    "like_rate", "comment_rate", "engagement_rate", "comment_like_ratio",
    "views_per_day", "likes_per_day", "comments_per_day",
    "views_growth_rate", "views_acceleration", "views_diff",
    "likes_diff", "engagement_diff",
    "view_count_log", "like_count_log", "comment_count_log",
    "subscriber_count_log", "views_to_subs_ratio", "views_per_day_per_sub",
    "channel_tier", "days_log", "age_bucket",
    "duration_log", "is_short_video",
    "snapshot_rank", "snapshot_count",
]

features = pd.DataFrame([{
    "like_rate": like_rate,
    "comment_rate": comment_rate,
    "engagement_rate": engagement_rate,
    "comment_like_ratio": comment_like_ratio,
    "views_per_day": views_per_day,
    "likes_per_day": likes_per_day,
    "comments_per_day": comments_per_day,
    "views_growth_rate": views_growth_rate,
    "views_acceleration": views_acceleration,
    "views_diff": views_diff,
    "likes_diff": likes_diff,
    "engagement_diff": engagement_diff,
    "view_count_log": view_count_log,
    "like_count_log": like_count_log,
    "comment_count_log": comment_count_log,
    "subscriber_count_log": subscriber_count_log,
    "views_to_subs_ratio": views_to_subs_ratio,
    "views_per_day_per_sub": views_per_day_per_sub,
    "channel_tier": channel_tier,
    "days_log": days_log,
    "age_bucket": age_bucket,
    "duration_log": duration_log,
    "is_short_video": is_short_video,
    "snapshot_rank": snapshot_rank,
    "snapshot_count": snapshot_count,
}])[FEATURE_COLUMNS]

# ─────────────────────────────────────────────
# PREDICT BUTTON — everything dependent on the prediction lives INSIDE this block
# ─────────────────────────────────────────────
if st.button("🔮 Predict Virality", type="primary"):
    probability = model.predict_proba(features)[0, 1]
    is_viral = probability >= optimal_threshold

    # Stash in session_state so the download button below (outside this block)
    # can still access the latest result after rerun
    st.session_state["last_probability"] = probability
    st.session_state["last_is_viral"] = is_viral

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Viral Probability", f"{probability:.1%}")
    with m2:
        st.metric("Prediction", "✅ VIRAL" if is_viral else "❌ Not Viral")

    if probability >= 0.7:
        st.success("High confidence — this video has strong viral signals!")
    elif probability >= 0.4:
        st.warning("Moderate — could go either way")
    else:
        st.info("Low confidence — this video likely lacks viral signals")

    with st.expander("🔍 See What the Model Saw"):
        st.write("Here's the feature values that were fed into the model:")

        st.write("**Engagement signals:**")
        st.write(f"- Like Rate: {like_rate*100:.2f}%")
        st.write(f"- Comment Rate: {comment_rate*100:.2f}%")
        st.write(f"- Engagement Rate: {engagement_rate*100:.2f}%")

        st.write("**Momentum Signals:**")
        st.write(f"- Views Per Day: {views_per_day:,.0f}")
        st.write(f"- Likes Per Day: {likes_per_day:,.0f}")
        st.write(f"- Comments Per Day: {comments_per_day:,.0f}")
        st.caption("No prior snapshot was provided, so growth/acceleration features were set to 0.")

        st.write("**Channel Context:**")
        tier_labels = ['Nano (<10K)', 'Micro (10K-100K)', 'Mid (100K-1M)', 'Macro (1M-10M)', 'Mega (10M+)']
        st.write(f"- Channel Tier: {tier_labels[channel_tier]}")
        st.write(f"- Views-to-Subs Ratio: {views_to_subs_ratio:.2f}x")

        st.write("**Full feature vector sent to the model:**")
        st.dataframe(features.T.rename(columns={0: "value"}))

else:
    st.caption("👆 Fill in the stats above and click **Predict Virality** to see results.")

# ─────────────────────────────────────────────
# COMPARISON TABLE (always visible — reflects current inputs)
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
# DOWNLOAD REPORT — uses last prediction made, if any
# ─────────────────────────────────────────────
if "last_probability" in st.session_state:
    probability = st.session_state["last_probability"]
    is_viral = st.session_state["last_is_viral"]

    result_summary = f"""Video Analysis Results
=====================
Views: {views:,}
Likes: {likes:,}
Comments: {comments:,}
Subscribers: {subscribers:,}
Days Old: {days_old}

Viral Probability: {probability:.1%}
Prediction: {"VIRAL ✅" if is_viral else "Not Viral ❌"}
Optimal Threshold: {optimal_threshold:.4f}
"""

    st.download_button(
        label="📥 Download Report",
        data=result_summary,
        file_name="virality_analysis.txt",
        mime="text/plain",
    )
else:
    st.caption("Run a prediction first to enable the report download.")

# ─────────────────────────────────────────────
# SIDEBAR — About
# ─────────────────────────────────────────────
st.sidebar.header("ℹ️ About This Model")
st.sidebar.write(f"""
**Prediction Model (Momentum Stability)**
- Algorithm: {threshold_info.get('model_name', 'XGBoost')}
- Features: 25 engineered signals
- Optimal threshold: {optimal_threshold:.4f}

Predicts whether a video's *current growth trajectory*
is likely to sustain viral performance in the near term —
not a long-range "will this ever go viral" forecast.
""")

st.sidebar.write("Built by Shayan Bhattacharjee")
st.sidebar.write("[GitHub](https://github.com/bhattacharjeeshayan860-netizen)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/shayan-bhattacharjee-860-netizen)")