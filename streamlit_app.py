import streamlit as st
import joblib
import pandas as pd

# Utility imports from your project
from utils.text_cleaner import clean_text
from utils.news_api_handler import extract_article_from_url, fetch_latest_headlines
from utils.language_handler import detect_and_translate_to_en

# Load models
@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    lr = joblib.load("models/logistic_model.pkl")
    gb = joblib.load("models/gb_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    return tfidf, lr, gb, rf

tfidf, lr_model, gb_model, rf_model = load_models()

def predict_news(text):
    """Preprocess text → Translate → Predict → Ensemble"""
    if not text or text.strip() == "":
        return "No input provided", {}

    # Preprocessing
    translated = detect_and_translate_to_en(text)
    cleaned = clean_text(translated)
    X = tfidf.transform([cleaned])

    models = {
        "Logistic Regression": lr_model,
        "Gradient Boosting": gb_model,
        "Random Forest": rf_model
    }

    votes = {}
    for name, model in models.items():
        preds = model.predict(X)[0]
        votes[name] = "Real" if preds == 0 else "Fake"

    # Ensemble
    majority_vote = max(set(votes.values()), key=list(votes.values()).count)

    return majority_vote, votes


# -------------------------------- UI -------------------------------- #

st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI — Fake News Detection using AI")

st.sidebar.title("Input Options")
choice = st.sidebar.radio("Choose input type:",
                          ["Text Input", "URL Input", "Live News"])

# ---------------- TEXT INPUT ----------------
if choice == "Text Input":
    st.subheader("Paste News Text / Headline")
    text = st.text_area("Enter text here:", height=180)

    if st.button("🔍 Analyze News"):
        result, votes = predict_news(text)
        st.success(f"✅ Prediction: **{result}**")
        st.write("### 🔎 Model Votes")
        st.json(votes)

# ---------------- URL INPUT ----------------
elif choice == "URL Input":
    st.subheader("Paste a News URL")
    url = st.text_input("Enter URL:")

    if st.button("🌐 Fetch & Analyze Article"):
        with st.spinner("Extracting article..."):
            article = extract_article_from_url(url)

        if article:
            st.write("### ✅ Extracted Text")
            st.write(article)

            result, votes = predict_news(article)
            st.success(f"✅ Prediction: **{result}**")
            st.write("### 🔎 Model Votes")
            st.json(votes)
        else:
            st.error("❌ Could not extract content from the URL.")

# ---------------- LIVE NEWS ----------------
elif choice == "Live News":
    st.subheader("🗞 Latest Headlines (API)")
    if st.button("📡 Fetch Live Headlines"):
        headlines = fetch_latest_headlines()

        if headlines:
            for i, news in enumerate(headlines):
                st.write(f"### 📰 {i+1}. {news}")

                result, votes = predict_news(news)
                st.info(f"Prediction: **{result}**")
        else:
            st.warning("⚠ Unable to fetch live news. Check API key or network.")

st.markdown("---")
st.caption("Built by NewsTruth AI | Fake News Detection Using Hybrid ML + NLP")
