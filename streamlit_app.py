import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load dataset
@st.cache_data
def load_news_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "news_data.csv")
    return pd.read_csv(data_path)

df = load_news_data()

# Search similar articles from dataset
def search_similar_articles(news_text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = df['News'].tolist() + [news_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = cosine_sim.argsort()[0, -top_n:][::-1]
    similar = df.iloc[top_indices][['News', 'Status']].to_dict(orient='records')
    return similar

# Google Custom Search API integration
def google_search(query, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': CSE_ID,
        'q': query,
        'num': num_results
    }
    response = requests.get(search_url, params=params)
    results = response.json().get('items', [])
    return results

# Streamlit UI
st.title("AI-Powered Fake News Detector")

# Input text area
news_text = st.text_area("Enter news article:", height=200)

if st.button("Analyze"):
    if news_text:
        with st.spinner("Analyzing news article..."):
            # Get similar articles
            similar_articles = search_similar_articles(news_text)
            
            # Get Google search results
            google_results = google_search(news_text)
            
            # Get OpenAI analysis
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a news authenticity analyzer. Analyze the given news article and determine its authenticity."},
                        {"role": "user", "content": f"Analyze this news article: {news_text}"}
                    ]
                )
                ai_analysis = response.choices[0].message.content
            except Exception as e:
                ai_analysis = f"Error getting AI analysis: {str(e)}"

        # Display results
        st.header("Analysis Results")
        
        st.subheader("AI Analysis")
        st.write(ai_analysis)
        
        st.subheader("Similar Articles from Dataset")
        for article in similar_articles:
            st.write(f"- {article['News']} (Status: {article['Status']})")
        
        st.subheader("Google Search Results")
        for result in google_results:
            st.write(f"- {result.get('title', '')}")
            st.write(f"  {result.get('snippet', '')}")
            st.write(f"  [Link]({result.get('link', '')})")
            st.write("---")
    else:
        st.warning("Please enter a news article to analyze.")
