import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Load the trained model
model=joblib.load("models/trained/detection_model.pkl")
threshold_info=joblib.load("models/trained/detection_threshold.pkl")
optimal_threshold=threshold_info['threshold']


st.title("🎵 Music Virality Detector")
st.write("Analyze if a music video is **currently viral** based on engagement signals")
st.info("💡 Tip: Get these stats from any YouTube video page")

st.subheader("📊 Video Statistics")
col1,col2=st.columns(2)
with col1:
    views=st.number_input("👁️ View Count", min_value=0, value=500000, step=10000, key="views")
    likes=st.number_input("👍 Like Count", min_value=0, value=25000, step=1000, key="likes")
    comments=st.number_input("💬 Comment Count", min_value=0, value=500, step=10, key="comments")
with col2:
    subscribers=st.number_input("👥 Subscriber Count", min_value=0, value=1000, step=100, key="subscribers")
    days_old=st.number_input("📅 Days Old", min_value=1, value=7, step=1, key="days_old")
duration_seconds=st.number_input("🕒 Duration (seconds)", min_value=10, value=60, key="duration_seconds")

# Derived metrics (computed every rerun so they are always defined)
like_rate=likes/views if views>0 else 0.0
comment_rate=comments/views if views>0 else 0.0
engagement_rate=(likes + 2 * comments)/views if views>0 else 0.0
views_per_day=views/days_old
likes_per_day=likes/days_old
comments_per_day=comments/days_old

comment_like_ratio=comments/likes if likes>0 else 0.0
views_growth_rate=0
views_acceleration=0
views_diff=0
likes_diff=0
engagement_diff=0
subscriber_count_log=np.log1p(subscribers)
views_to_subs_ratio=views/subscribers if subscribers>0 else 0.0
views_per_day_per_sub=(views_per_day/subscribers) if subscribers>0 else 0.0

if subscribers<10_000:
    channel_tier = 0
elif subscribers<100_000:
    channel_tier=1
elif subscribers<1_000_000:
    channel_tier=2
elif subscribers < 10_000_000:
    channel_tier = 3
else:
    channel_tier=4

days_log=np.log1p(days_old)
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

duration_log=np.log1p(duration_seconds)
is_short_video= 1.0 if duration_seconds<60 else 0.0

if st.button("Predict Virality"):
    # Build the DataFrame
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
    "subscriber_count_log": subscriber_count_log,
    "views_to_subs_ratio": views_to_subs_ratio,
    "views_per_day_per_sub": views_per_day_per_sub,
    "channel_tier": channel_tier,
    "days_log": days_log,
    "age_bucket": age_bucket,
    "duration_log": duration_log,
    "is_short_video": is_short_video,
    "snapshot_rank": 1,
    "snapshot_count": 1
}])
    proba = model.predict_proba(features)[0, 1]
    is_viral = proba >= optimal_threshold
    st.metric("Viral probability", f"{proba:.1%}")
    st.metric("Prediction", "✅ VIRAL" if is_viral else "❌ Not Viral")
    if proba >= 0.7:
        st.success("High confidence — this video has strong viral signals!")
    elif proba >= 0.4:
        st.warning("Moderate — could go either way")
    else:
        st.info("Low confidence — this video likely lacks viral signals")
with st.expander("🔍 See What the Model Saw"):
    st.write("**Engagement signals:**")
    st.write(f"- Like Rate: {like_rate*100:.2f}%")
    st.write(f"- Comment Rate: {comment_rate*100:.2f}%")

    st.write(f"**Momentum Signals:**")
    st.write(f"- Views Per Day: {views_per_day:,.0f}")
    st.write(f"- Likes Per Day: {likes_per_day:,.0f}")
    st.write(f"- Comments Per Day: {comments_per_day:,.0f}")

    st.write(f"**Channel Context:**")
    st.write(f"-Channel Tier: {['Nano (<10K)', 'Micro (10K-100K)', 'Mid (100K-1M)', 'Macro (1M-10M)', 'Mega (10M+)'][channel_tier]}")
    st.write(f"views-to-subs-ratio: {views_to_subs_ratio:.2f}x")

st.subheader("📈 How Does This Compare?")
benchmark=pd.DataFrame({
                        "Metric":["views/day","Likes Rate","Engagement Rate","Channel Size"],
                        "This video":[
                            f"{views_per_day:,.0f}",
                            f"{like_rate*100:.1f}%",
                            f"{engagement_rate*100:.1f}%",
                            f"{subscribers:,}"
                        ],
                        "Viral Typical":["50k+ /day",
                                         "4-8%",
                                         "3-6%",
                                         "100k+ subs"]
                                         })
st.table(benchmark)

st.sidebar.header("🎬 Try Sample Videos")


def _set_sample_video(values):
    for key, value in values.items():
        st.session_state[key] = value

st.sidebar.button(
    "💥 Viral Hit Example",
    on_click=_set_sample_video,
    kwargs={
        "values": {
            "views": 5000000,
            "likes": 250000,
            "comments": 15000,
            "subscribers": 500000,
            "days_old": 7,
            "duration_seconds": 180,
        }
    },
)

st.sidebar.button(
    "📉 Struggling Video Example",
    on_click=_set_sample_video,
    kwargs={
        "values": {
            "views": 50000,
            "likes": 1000,
            "comments": 50,
            "subscribers": 5000,
            "days_old": 14,
            "duration_seconds": 200,
        }
    },
)