import streamlit as st
import joblib
import re
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator

# -------------------- Clean Text --------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = text.lower().strip()
    return text

# -------------------- Language Detect + Translate --------------------
def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

# -------------------- URL to Text --------------------
def extract_article(url):
    # Try newspaper3k
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return article.text
    except:
        pass

    # Fallback to BeautifulSoup
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ").strip()
    except:
        return ""

# -------------------- Live News API --------------------
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # add your key

def fetch_headlines():
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language=en"
        res = requests.get(url).json()
        return [n["title"] for n in res.get("results", [])]
    except:
        return []

# -------------------- Load Models (Root Folder) --------------------
@st.cache_resource
def load_all_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    lr = joblib.load("logistic_model.pkl")
    gb = joblib.load("gb_model.pkl")
    rf = joblib.load("rf_model.pkl")
    return tfidf, lr, gb, rf

tfidf, lr_model, gb_model, rf_model = load_all_models()

# -------------------- Predict --------------------
def predict_news(text):
    text = detect_and_translate(text)
    text = clean_text(text)
    vector = tfidf.transform([text])

    models = {
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model,
        "Random Forest": rf_model
    }

    votes = {name: ("Real" if m.predict(vector)[0] == 0 else "Fake") for name, m in models.items()}
    final_result = max(set(votes.values()), key=list(votes.values()).count)

    return final_result, votes

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI — Fake News Detection")

mode = st.sidebar.radio("Choose Input Type", ["Text Input", "URL Input", "Live News"])

# Text Input
if mode == "Text Input":
    text = st.text_area("Enter text or headline:")
    if st.button("Analyze"):
        result, votes = predict_news(text)
        st.success(f"Prediction: **{result}**")
        st.json(votes)

# URL Input
elif mode == "URL Input":
    url = st.text_input("Paste news article URL:")
    if st.button("Analyze URL"):
        with st.spinner("Extracting article..."):
            article = extract_article(url)
        st.write(article[:1000] + "...")
        result, votes = predict_news(article)
        st.success(f"Prediction: **{result}**")
        st.json(votes)

# Live News
elif mode == "Live News":
    if st.button("Fetch Headlines"):
        headlines = fetch_headlines()
        if not headlines:
            st.warning("No news fetched. Check API key.")
        for idx, h in enumerate(headlines):
            st.write(f"### {idx+1}. {h}")
            result, votes = predict_news(h)
            st.info(f"Prediction: **{result}**")
