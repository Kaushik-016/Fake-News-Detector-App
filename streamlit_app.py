import streamlit as st
import joblib
import re
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator

# -------------------- Text Cleaning --------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = text.lower().strip()
    return text

# -------------------- Language Detection + Translation --------------------
def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

# -------------------- URL Article Extraction --------------------
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
        text = soup.get_text(separator=" ")
        return text.strip()
    except:
        return ""

# -------------------- Live News API --------------------
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Replace with your key

def fetch_headlines():
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language=en"
        res = requests.get(url).json()
        return [a["title"] for a in res.get("results", [])]
    except:
        return []

# -------------------- Load Models --------------------
@st.cache_resource
def load_all_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    lr = joblib.load("models/logistic_model.pkl")
    gb = joblib.load("models/gb_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    return tfidf, lr, gb, rf

tfidf, lr_model, gb_model, rf_model = load_all_models()

# -------------------- Prediction Function --------------------
def predict_news(text):
    text = detect_and_translate(text)
    text = clean_text(text)
    X = tfidf.transform([text])

    models = {
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model,
        "Random Forest": rf_model
    }

    votes = {name: ("Real" if m.predict(X)[0] == 0 else "Fake") for name, m in models.items()}
    final = max(set(votes.values()), key=list(votes.values()).count)
    return final, votes

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI — Fake News Detector")

option = st.sidebar.radio("Select Input Type", ["Text Input", "URL Input", "Live News"])

# -------- Text Input --------
if option == "Text Input":
    text = st.text_area("Enter News Text or Headline:", height=170)
    if st.button("🔍 Analyze"):
        result, votes = predict_news(text)
        st.success(f"✅ Prediction: **{result}**")
        st.write("### Model Votes")
        st.json(votes)

# -------- URL Input --------
elif option == "URL Input":
    url = st.text_input("Paste News Article URL:")
    if st.button("🌐 Fetch & Analyze"):
        with st.spinner("Extracting article text..."):
            article = extract_article(url)

        if article:
            st.write("### ✅ Extracted Text:")
            st.write(article[:1500] + " ...")
            result, votes = predict_news(article)
            st.success(f"✅ Prediction: **{result}**")
            st.json(votes)
        else:
            st.error("❌ Could not extract article.")

# -------- Live News --------
elif option == "Live News":
    if st.button("🕵 Fetch Latest Headlines"):
        headlines = fetch_headlines()
        if not headlines:
            st.warning("⚠ Couldn't fetch news. Check API key or internet.")
        for i, h in enumerate(headlines):
            st.write(f"### {i+1}. {h}")
            result, votes = predict_news(h)
            st.info(f"Prediction: **{result}**")
