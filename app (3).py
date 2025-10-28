import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# from bertopic.topic_info import Name # This import is not needed if we get Name from topic_model.get_topic_info()

# --- NLTK Downloads ---
import nltk
import os

# Define a directory to download NLTK data in the app's root directory
nltk_data_path = os.path.join(".", "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip', paths=[nltk_data_path])
except LookupError:
    try:
        nltk.download('vader_lexicon', download_dir=nltk_data_path)
    except Exception as e:
        st.error(f"Failed to download NLTK vader_lexicon: {e}")

try:
    nltk.data.find('tokenizers/punkt.zip', paths=[nltk_data_path])
except LookupError:
     try:
         nltk.download('punkt', download_dir=nltk_data_path)
     except Exception as e:
         st.error(f"Failed to download NLTK punkt: {e}")


# --- Load Data ---
st.set_page_config(page_title="Shopee Descriptive Analysis Dashboard", layout="wide")
st.title("🛒 Shopee Review Descriptive Analysis Dashboard")

# Load data directly from the CSV file
df_full = pd.read_csv('Shopee_translated.csv', delimiter=';', on_bad_lines='skip')
df = df_full.head(1000).copy() # Using head(1000) and .copy() to match the notebook's data

# --- Perform Sentiment Analysis (replicated from notebook for self-containment) ---
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


# Load RoBERTa model and tokenizer (cached)
@st.cache_resource
def load_roberta_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_roberta_model()

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# Calculate sentiment scores and merge with dataframe
sia = SentimentIntensityAnalyzer()
res = {}
for i, row in df.iterrows(): # Iterate through df to match the notebook's analysis on head(1000)
    try:
        text = str(row['content_en']) # Ensure text is string
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        # Handle potential RuntimeError from tokenizer/model if needed, or log
        pass # Ignoring for simplicity in app

sentiment_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
df = df.merge(sentiment_df, how='left')


# --- Perform Topic Modeling (replicated from notebook for self-containment) ---
from bertopic import BERTopic

# Ensure NLTK punkt tokenizer is available for BERTopic if needed (though BERTopic often uses SentenceTransformers directly)
# Try downloading punkt, but pass if already exists or fails
# This check and download is now handled in the NLTK Downloads section at the top


# BERTopic needs a list of documents; use 'content_en' from the potentially filtered df
abstracts = df['content_en'].to_list()

@st.cache_resource # Cache the BERTopic model
def fit_bertopic_model(docs):
    # Use a pre-trained sentence transformer model
    from sentence_transformers import SentenceTransformer
    # Explicitly set a common model that is likely compatible
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    topic_model = BERTopic() # You can pass min_topic_size here if needed
    topics, probs = topic_model.fit_transform(docs, embeddings) # Fit with precomputed embeddings
    return topic_model, topics

# Fit the model to the abstracts from the loaded data
topic_model, topic = fit_bertopic_model(abstracts)

# Add topic information to the DataFrame
df['topic'] = topic

# Get topic names from the fitted model
topic_info_df = topic_model.get_topic_info()
# Create a mapping from topic id to topic name
topic_name_map = topic_info_df.set_index('Topic')['Name'].to_dict()
# Map the topic ids in df to topic names
df['topic_name'] = df['topic'].map(topic_name_map)


# Ensure sentiment_label and review_en columns exist for plotting/wordcloud
if 'sentiment_label' not in df.columns:
    def get_sentiment_label(row):
        # Use Roberta scores for sentiment label
        if row['roberta_pos'] > row['roberta_neg'] and row['roberta_pos'] > row['roberta_neu']:
            return 'positive'
        elif row['roberta_neg'] > row['roberta_pos'] and row['roberta_neg'] > row['roberta_neu']:
            return 'negative'
        else:
            return 'neutral'
    df['sentiment_label'] = df.apply(get_sentiment_label, axis=1)

if 'review_en' not in df.columns:
    df['review_en'] = df['content_en']

# The rest of the Streamlit app code remains the same, operating on the 'df' DataFrame

# --- Sidebar Filters ---
# Ensure -1 (outlier topic) is an option
all_topics = ['-1_the_to_and_is'] + sorted([name for name in df['topic_name'].unique() if name != '-1_the_to_and_is'])
topics = st.sidebar.multiselect("Select Topics", all_topics, default=all_topics)

if topics:
    df_filtered = df[df['topic_name'].isin(topics)]
else:
    df_filtered = df # Use all data if no topics are selected

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
if not df_filtered.empty:
    sent_count = df_filtered['sentiment_label'].value_counts().reset_index()
    sent_count.columns = ['Sentiment', 'Count']
    fig1 = px.pie(sent_count, values='Count', names='Sentiment', color='Sentiment',
                  color_discrete_map={'positive':'green','neutral':'gray','negative':'red'})
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.write("No data available for selected topics.")


# --- Topic Frequency ---
st.subheader("Top Topics by Frequency")
if not df_filtered.empty:
    # Group by topic_name to get counts of selected topics
    topic_freq = df_filtered.groupby('topic_name').size().reset_index(name='Count').sort_values('Count', ascending=False)
    # Include topics that were selected but have 0 count in the filtered data
    selected_topics_with_zero_count = pd.DataFrame({'topic_name': topics, 'Count': 0})
    topic_freq = pd.concat([topic_freq, selected_topics_with_zero_count]).drop_duplicates(subset=['topic_name']).sort_values('Count', ascending=False)
    # Filter out the -1 topic from this visualization unless specifically selected and has count > 0
    if '-1_the_to_and_is' in topic_freq['topic_name'].values and '-1_the_to_and_is' not in topics and topic_freq[topic_freq['topic_name'] == '-1_the_to_and_is']['Count'].iloc[0] > 0:
         # If -1 is in the data but not explicitly selected, and has reviews, keep it
         pass # Keep it for now, filtering happens in the `if topics:` block
    elif '-1_the_to_and_is' in topic_freq['topic_name'].values and '-1_the_to_and_is' in topics and topic_freq[topic_freq['topic_name'] == '-1_the_to_and_is']['Count'].iloc[0] == 0:
        # If -1 was selected but has 0 reviews in filtered data, remove it
        topic_freq = topic_freq[topic_freq['topic_name'] != '-1_the_to_and_is']
    elif '-1_the_to_and_is' in topic_freq['topic_name'].values and '-1_the_to_and_is' not in topics:
         # If -1 is in the data but not explicitly selected, remove it for this chart
        topic_freq = topic_freq[topic_freq['topic_name'] != '-1_the_to_and_is']


    fig2 = px.bar(topic_freq, x='Count', y='topic_name', orientation='h', color='Count',
                  color_continuous_scale='Blues', title="Frequency of Selected Topics (Excluding Outliers by Default)")
    st.plotly_chart(fig2, use_container_width=True)
else:
     st.write("No data available for selected topics.")

# --- Average Sentiment by Topic ---
st.subheader("Average Negative Sentiment per Topic")
if not df_filtered.empty:
    # Group by topic_name for selected topics only
    topic_sent = df_filtered.groupby('topic_name')['roberta_neg'].mean().reset_index().sort_values('roberta_neg', ascending=False)

    fig3 = px.bar(topic_sent, x='roberta_neg', y='topic_name', orientation='h', color='roberta_neg',
                  color_continuous_scale='Reds', labels={'roberta_neg': 'Avg Negativity'},
                  title="Average Roberta Negative Sentiment for Selected Topics")
    st.plotly_chart(fig3, use_container_width=True)
else:
     st.write("No data available for selected topics.")

# --- Word Cloud (Keywords Visualization) ---
st.subheader("Word Cloud of All Reviews in Selected Topics")
if not df_filtered.empty:
    text = " ".join(review for review in df_filtered.review_en.astype(str))
    if text.strip(): # Check if there is any text to generate wordcloud
        wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No text available for Word Cloud in selected topics.")
else:
     st.write("No data available for selected topics.")
