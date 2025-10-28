import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic.topic_info import Name

# --- Load Data ---
st.set_page_config(page_title="Shopee Descriptive Analysis Dashboard", layout="wide")
st.title("🛒 Shopee Review Descriptive Analysis Dashboard")

# Use the existing results_df DataFrame
# Ensure necessary columns exist or are created
if 'topic_name' not in results_df.columns:
    # Get topic names from topic_model.get_topic_info()
    topic_info_df = topic_model.get_topic_info()
    # Create a mapping from topic id to topic name
    topic_name_map = topic_info_df.set_index('Topic')['Name'].to_dict()
    # Map the topic ids in results_df to topic names
    results_df['topic_name'] = results_df['topic'].map(topic_name_map)


if 'sentiment_label' not in results_df.columns:
    def get_sentiment_label(row):
        if row['roberta_pos'] > row['roberta_neg'] and row['roberta_pos'] > row['roberta_neu']:
            return 'positive'
        elif row['roberta_neg'] > row['roberta_pos'] and row['roberta_neg'] > row['roberta_neu']:
            return 'negative'
        else:
            return 'neutral'
    results_df['sentiment_label'] = results_df.apply(get_sentiment_label, axis=1)

if 'review_en' not in results_df.columns:
    results_df['review_en'] = results_df['content_en']

df_app = results_df.copy()

# --- Sidebar Filters ---
topics = st.sidebar.multiselect("Select Topics", df_app['topic_name'].unique())
if topics:
    df_app = df_app[df_app['topic_name'].isin(topics)]

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
sent_count = df_app['sentiment_label'].value_counts().reset_index()
sent_count.columns = ['Sentiment', 'Count']
fig1 = px.pie(sent_count, values='Count', names='Sentiment', color='Sentiment',
              color_discrete_map={'positive':'green','neutral':'gray','negative':'red'})
st.plotly_chart(fig1, use_container_width=True)

# --- Topic Frequency ---
st.subheader("Top Topics by Frequency")
topic_freq = df_app.groupby('topic_name').size().reset_index(name='Count').sort_values('Count', ascending=False)
fig2 = px.bar(topic_freq, x='Count', y='topic_name', orientation='h', color='Count',
              color_continuous_scale='Blues')
st.plotly_chart(fig2, use_container_width=True)

# --- Average Sentiment by Topic ---
st.subheader("Average Negative Sentiment per Topic")
topic_sent = df_app.groupby('topic_name')['roberta_neg'].mean().reset_index().sort_values('roberta_neg', ascending=False)
fig3 = px.bar(topic_sent, x='roberta_neg', y='topic_name', orientation='h', color='roberta_neg',
              color_continuous_scale='Reds', labels={'roberta_neg': 'Avg Negativity'})
st.plotly_chart(fig3, use_container_width=True)

# --- Word Cloud (Keywords Visualization) ---
st.subheader("Word Cloud of All Reviews")
text = " ".join(review for review in df_app.review_en.astype(str))
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)
