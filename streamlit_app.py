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

# -------------------- Detect + Translate --------------------
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
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            return article.text
    except:
        pass

    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ").strip()
    except:
        return ""

# -------------------- Live News Fetch --------------------
NEWS_API_KEY = "YOUR_NEWS_API_KEY"

def fetch_headlines():
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language=en"
        res = requests.get(url).json()
        return [a["title"] for a in res.get("results", [])]
    except:
        return []

# -------------------- Load Models (YOUR filenames) --------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidfvect.pkl")   # ✅ matches your repo
    model1 = joblib.load("model.pkl")      # ✅ main model
    model2 = joblib.load("model2.pkl")     # ✅ second model
    return tfidf, model1, model2

tfidf, model1, model2 = load_models()

# -------------------- Predict --------------------
def predict_news(text):
    text = detect_and_translate(text)
    text = clean_text(text)
    vec = tfidf.transform([text])

    predictions = {
        "Model 1 (model.pkl)": "Real" if model1.predict(vec)[0] == 0 else "Fake",
        "Model 2 (model2.pkl)": "Real" if model2.predict(vec)[0] == 0 else "Fake"
    }

    final = max(set(predictions.values()), key=list(predictions.values()).count)
    return final, predictions

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI — Fake News Detector")

mode = st.sidebar.radio("Select Input Type", ["Text Input", "URL Input", "Live News"])

if mode == "Text Input":
    text = st.text_area("Enter News Text")
    if st.button("Analyze"):
        result, votes = predict_news(text)
        st.success(f"Prediction: **{result}**")
        st.write("Votes:")
        st.json(votes)

elif mode == "URL Input":
    url = st.text_input("Enter article URL")
    if st.button("Analyze URL"):
        with st.spinner("Extracting article..."):
            article = extract_article(url)
        st.write(article[:1500] + "...")
        result, votes = predict_news(article)
        st.success(f"Prediction: **{result}**")
        st.json(votes)

elif mode == "Live News":
    if st.button("Fetch Live Headlines"):
        headlines = fetch_headlines()
        for i, h in enumerate(headlines):
            st.write(f"### {i+1}. {h}")
            result, votes = predict_news(h)
            st.info(f"Prediction: **{result}**")
