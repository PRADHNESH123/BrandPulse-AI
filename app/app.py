import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import json
import sys
import time
import random
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append('C:/data/BrandPulse-AI')
from src.preprocessing import clean_tweet

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title='BrandPulse AI',
    page_icon='📊',
    layout='wide'
)

# ── Load models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf    = joblib.load('C:/data/BrandPulse-AI/models/tfidf_vectorizer.pkl')
    lr_model = joblib.load('C:/data/BrandPulse-AI/models/logistic_model.pkl')
    lstm     = load_model('C:/data/BrandPulse-AI/models/lstm_best.h5')
    with open('C:/data/BrandPulse-AI/models/tokenizer.json') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    return tfidf, lr_model, lstm, tokenizer

tfidf, lr_model, lstm_model, tokenizer = load_models()

MAX_LEN = 100

# ── Prediction function ───────────────────────────────────
def predict_sentiment(text, model_choice):
    cleaned = clean_tweet(text)
    if model_choice == 'Logistic Regression':
        vec  = tfidf.transform([cleaned])
        pred = lr_model.predict(vec)[0]
        conf = max(lr_model.predict_proba(vec)[0]) * 100
    else:
        seq  = tokenizer.texts_to_sequences([cleaned])
        pad  = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prob = lstm_model.predict(pad, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0
        conf = max(prob, 1 - prob) * 100
    return pred, conf, cleaned

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.image('https://img.icons8.com/color/96/000000/bar-chart.png', width=80)
st.sidebar.title('BrandPulse AI')
st.sidebar.markdown('Twitter Sentiment Analysis')
st.sidebar.divider()

model_choice = st.sidebar.selectbox(
    'Choose Model',
    ['Logistic Regression', 'LSTM (BiLSTM)']
)

st.sidebar.divider()
st.sidebar.markdown('### Model Performance')
st.sidebar.metric('Logistic Regression', '~80%', 'Accuracy')
st.sidebar.metric('LSTM (BiLSTM)', '81.04%', 'Accuracy')

# ── Header ────────────────────────────────────────────────
st.title('📊 BrandPulse AI — Twitter Sentiment Analysis')
st.markdown('Real-time sentiment analysis powered by Classical NLP and Deep Learning')
st.divider()

# ── Tab layout ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🔍 Live Predictor',
                             '📈 Trend Analysis',
                             '⚖️ Model Comparison'])

# ════════════════════════════════════════════════════════
# TAB 1 — Live Predictor
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader(' Predict Tweet Sentiment')

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            'Enter a tweet:',
            placeholder='Type your tweet here...',
            height=120
        )

        if st.button('Analyze Sentiment ', type='primary'):
            if user_input.strip():
                with st.spinner('Analyzing...'):
                    pred, conf, cleaned = predict_sentiment(
                        user_input, model_choice)

                if pred == 1:
                    st.success(f'😊 POSITIVE — Confidence: {conf:.1f}%')
                else:
                    st.error(f'😞 NEGATIVE — Confidence: {conf:.1f}%')

                with st.expander('See cleaned text'):
                    st.write(f'**Cleaned:** {cleaned}')
            else:
                st.warning('Please enter a tweet!')

    with col2:
        st.markdown('### Try these examples:')
        examples = [
            "This product is absolutely amazing!",
            "Worst experience ever. Never again!",
            "Not happy with the service today.",
            "Best purchase I have ever made!",
            "The delivery was late and damaged."
        ]
        for ex in examples:
            if st.button(ex[:40] + '...', key=ex):
                pred, conf, cleaned = predict_sentiment(ex, model_choice)
                if pred == 1:
                    st.success(f'😊 POSITIVE ({conf:.1f}%)')
                else:
                    st.error(f'😞 NEGATIVE ({conf:.1f}%)')

    st.divider()

    # Batch prediction
    st.subheader('📋 Batch Prediction')
    batch_input = st.text_area(
        'Enter multiple tweets (one per line):',
        height=150,
        placeholder='Tweet 1\nTweet 2\nTweet 3'
    )

    if st.button('Analyze All '):
        if batch_input.strip():
            tweets = [t.strip() for t in batch_input.split('\n') if t.strip()]
            results = []
            for tweet in tweets:
                pred, conf, cleaned = predict_sentiment(tweet, model_choice)
                results.append({
                    'Tweet': tweet[:60] + '...' if len(tweet) > 60 else tweet,
                    'Sentiment': '😊 Positive' if pred == 1 else '😞 Negative',
                    'Confidence': f'{conf:.1f}%'
                })
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            pos = sum(1 for r in results if 'Positive' in r['Sentiment'])
            neg = len(results) - pos
            fig = px.pie(
                values=[pos, neg],
                names=['Positive', 'Negative'],
                color_discrete_map={'Positive': '#4ECDC4', 'Negative': '#FF6B6B'},
                title='Batch Sentiment Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Trend Analysis
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader('📈 Sentiment Trend — Last 24 Hours')

    # Simulate 24hr trend data
    hours = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    np.random.seed(42)
    positive_counts = np.random.randint(40, 80, 24)
    negative_counts = 100 - positive_counts

    df_trend = pd.DataFrame({
        'Time': hours,
        'Positive': positive_counts,
        'Negative': negative_counts
    })

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_trend['Time'], y=df_trend['Positive'],
        name='Positive', fill='tozeroy',
        line=dict(color='#4ECDC4', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_trend['Time'], y=df_trend['Negative'],
        name='Negative', fill='tozeroy',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig_trend.update_layout(
        title='Sentiment Trend Over Last 24 Hours',
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Tweets', '2,400', '+12%')
    col2.metric('Positive', f'{positive_counts.mean():.0f}%', '+5%')
    col3.metric('Negative', f'{negative_counts.mean():.0f}%', '-5%')

    # Pie chart
    st.subheader('Overall Sentiment Distribution')
    total_pos = int(positive_counts.sum())
    total_neg = int(negative_counts.sum())

    fig_pie = px.pie(
        values=[total_pos, total_neg],
        names=['Positive', 'Negative'],
        color_discrete_map={'Positive': '#4ECDC4', 'Negative': '#FF6B6B'},
        hole=0.4
    )
    fig_pie.update_layout(title='24-Hour Sentiment Split')
    st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader('⚖️ Classical NLP vs Deep Learning')

    col1, col2, col3 = st.columns(3)
    col1.metric('Logistic Regression', '~80%', 'F1: 0.80')
    col2.metric('Naive Bayes', '~78%', 'F1: 0.78')
    col3.metric('LSTM (BiLSTM)', '81.04%', 'F1: 0.81')

    # Comparison chart
    comparison_data = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'LSTM (BiLSTM)'],
        'Accuracy': [0.80, 0.78, 0.8104],
        'F1-Score': [0.80, 0.78, 0.81],
        'Speed': ['Fast', 'Very Fast', 'Slow'],
        'Interpretable': ['Yes', 'Yes', 'No']
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True)

    fig_comp = px.bar(
        df_comp,
        x='Model',
        y=['Accuracy', 'F1-Score'],
        barmode='group',
        color_discrete_map={'Accuracy': '#4ECDC4', 'F1-Score': '#9B59B6'},
        title='Accuracy vs F1-Score Comparison'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    ### Key Takeaways
    - **LSTM** achieves highest accuracy by understanding word context
    - **Logistic Regression** is 10x faster and nearly as accurate
    - **Naive Bayes** is lightest but misses complex patterns
    - For production: use **Logistic Regression** for speed,
      **LSTM** for maximum accuracy
    """)