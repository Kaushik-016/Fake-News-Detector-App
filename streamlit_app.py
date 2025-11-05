import streamlit as st
import joblib
import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article
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

# -------------------- Extract Text From URL --------------------
def extract_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        except:
            return ""

# -------------------- Fetch Live News Headlines --------------------
NEWS_API_KEY = "YOUR_NEWS_API_KEY"   # ✅ Replace with your key
def fetch_headlines():
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language=en&category=top"
        res = requests.get(url).json()
        return [article["title"] for article in res.get("results", [])]
    except:
        return []

# -------------------- Load ML Models --------------------
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

    votes = {name: ("Real" if model.predict(X)[0] == 0 else "Fake") 
             for name, model in models.items()}

    final = max(set(votes.values()), key=list(votes.values()).count)
    return final, votes

# -------------------- UI --------------------
st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI — Fake News Detector")

option = st.sidebar.radio("Select Input Type", ["Text", "URL", "Live News"])

# -------- Text Input --------
if option == "Text":
    text = st.text_area("Enter News Text or Headline")
    if st.button("Analyze News"):
        result, votes = predict_news(text)
        st.success(f"Prediction: **{result}**")
        st.json(votes)

# -------- URL Input --------
elif option == "URL":
    url = st.text_input("Enter News Article URL")
    if st.button("Analyze URL"):
        with st.spinner("Extracting article text..."):
            article = extract_article(url)
        st.write("### Extracted Text:")
        st.write(article[:1500] + "...") if len(article) > 1500 else st.write(article)
        result, votes = predict_news(article)
        st.success(f"Prediction: **{result}**")
        st.json(votes)

# -------- Live News --------
elif option == "Live News":
    if st.button("Fetch Live Headlines"):
        headlines = fetch_headlines()
        for i, h in enumerate(headlines):
            st.write(f"### {i+1}. {h}")
            result, votes = predict_news(h)
            st.info(f"Prediction: **{result}**")
