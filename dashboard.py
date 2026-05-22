import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Load the trained model
model=joblib.load("models/trained/detection_model.pkl")
threshold_info=joblib.load("models/trained/detection_threshold.pkl")
optimal_threshold=threshold_info['threshold']


st.title("🎵 Music Virality Predictor")
st.write("Predict if a music video is viral based on engagement signals")


views=st.number_input("Enter the number of views",min_value=0,value=500000)
likes=st.number_input("Enter the number of likes",min_value=0,value=25000)
comments=st.number_input("Enter the number of comments",min_value=0,value=500)
subscribers=st.number_input("Enter the number of subscribers",min_value=0,value=1000)
days_old=st.number_input("Enter the number of days since upload",min_value=1,value=7)
duration_seconds=st.number_input("Enter the duration of the video in seconds",min_value=10,value=60)
if st.button("Predict Virality"):
    like_rate=likes/views if views>0 else 0
    comment_rate=comments/views if views>0 else 0
    engagement_rate=(likes+ 2 * comments)/views if views>0 else 0
    views_per_day=views/days_old 
    likes_per_day=likes/days_old
    comments_per_day=comments/days_old
    
    
    comment_like_ratio=comments/likes if likes>0 else 0
    views_growth_rate=0
    views_acceleration=0
    views_diff=0
    likes_diff=0
    engagement_diff=0
    subscriber_count_log=np.log1p(subscribers)
    views_to_subs_ratio=views/subscribers if subscribers>0 else 0
    views_per_day_per_sub=(views_per_day/subscribers) if subscribers>0 else 0
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