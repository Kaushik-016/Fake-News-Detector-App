import joblib
import requests
import pandas as pd
import re
import os
from flask import Flask, render_template, request, redirect, url_for
from goose3 import Goose
from forms import NewsForm
from config import NEWS_API_KEY, NEWS_API_URL
import numpy as np
from collections import Counter

# --- Application Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secure_key_for_dev')

# --- Model loading: load ALL models ---
MODEL_NAMES = ['PassiveAggressive', 'SVC', 'MultinomialNB', 'LogisticRegression', 'RandomForest', 'XGBoost']

models = {}           # dict: name -> estimator
tfidfvect = None      # TfidfVectorizer

def load_models():
    global models, tfidfvect
    # Load vectorizer
    try:
        tfidfvect = joblib.load('tfidfvect.pkl')
        print("Loaded tfidfvect.pkl")
    except Exception as e:
        print("Error loading tfidfvect.pkl:", e)
        tfidfvect = None

    # Load each model file model_<Name>.pkl
    for name in MODEL_NAMES:
        path = f"model_{name}.pkl"
        try:
            m = joblib.load(path)
            models[name] = m
            print(f"Loaded {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

# call at startup
load_models()

# --- Text Cleaning Utility ---
def clean_input_text(text):
    """Normalizes input text for consistent training/prediction."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

# --- Helper Functions ---

def get_article_content(url):
    """Fetches the URL and extracts clean article text and title using Goose3."""
    try:
        g = Goose({'browser_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        article = g.extract(url=url)
        article_text = f"{article.title}. {article.cleaned_text}"
        if len(article_text) < 50 or not article.title:
            return None, None
        return article_text, article.title
    except Exception as e:
        print(f"Error extracting content from URL {url}: {e}")
        return None, None

def check_external_sources(title):
    """Searches external news sources using the NewsData.io API for corroboration."""
    if not NEWS_API_KEY or len(NEWS_API_KEY) < 20:
        return "API Check Skipped (API Key invalid or missing)."

    params = {
        'apikey': NEWS_API_KEY,
        'q': title,
        'language': 'en',
        'country': 'us,gb,in',
        'size': 5,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        total_results = data.get('totalResults', 0)
        if total_results > 5:
            return f"SUCCESS: {total_results} similar articles found on external APIs. (High Corroboration)"
        elif total_results > 0:
            return f"NOTICE: {total_results} similar articles found. (Low Corroboration)"
        else:
            return "WARNING: Zero similar articles found on external APIs. (Lacks Corroboration)"
    except requests.exceptions.RequestException as e:
        return f"API Check Failed: Network or API error. ({e})"

# --- New: Predict with all models and ensemble ---
def predict_all(news_text):
    """Return dict with each model's prediction and ensemble outputs."""
    result = {'per_model': {}, 'ensemble': {}}
    if not models or tfidfvect is None:
        result['error'] = "Model Error: ML components not loaded correctly."
        return result

    cleaned_text = clean_input_text(news_text)
    X_vect = tfidfvect.transform([cleaned_text])

    labels = []
    prob_maps = []  # list of dicts {'REAL': p, 'FAKE': q}

    for name, m in models.items():
        try:
            pred = m.predict(X_vect)[0]
            label = "REAL" if int(pred) == 1 else "FAKE"
            entry = {'prediction': label}
            labels.append(label)

            # Try predict_proba
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(X_vect)[0]
                classes = list(m.classes_)
                prob_map = {}
                for cls_idx, cls_val in enumerate(classes):
                    key_label = "REAL" if int(cls_val) == 1 else "FAKE"
                    prob_map[key_label] = float(probs[cls_idx])
                entry['proba'] = prob_map
                prob_maps.append(prob_map)
            elif hasattr(m, "decision_function"):
                # Convert decision_function output into softmax probabilities
                df = m.decision_function(X_vect)
                # handle different shapes
                if np.ndim(df) == 0 or (np.ndim(df) == 1 and df.shape[0] == 1):
                    score = float(df)
                    scores = np.array([-score, score])
                else:
                    scores = np.array(df[0])
                exp = np.exp(scores - np.max(scores))
                probs = exp / exp.sum()
                classes = getattr(m, "classes_", [0,1])
                prob_map = {}
                for idx, cls in enumerate(classes):
                    key_label = "REAL" if int(cls) == 1 else "FAKE"
                    prob_map[key_label] = float(probs[idx])
                entry['proba'] = prob_map
                prob_maps.append(prob_map)
            else:
                entry['proba'] = None

            result['per_model'][name] = entry
        except Exception as e:
            result['per_model'][name] = {'error': str(e)}

    # Ensemble: majority vote
    if labels:
        vote_count = Counter(labels)
        most_common_label, cnt = vote_count.most_common(1)[0]
        result['ensemble']['majority_vote'] = {'prediction': most_common_label, 'counts': dict(vote_count)}
    else:
        result['ensemble']['majority_vote'] = {'note': 'No model predictions available.'}

    # Ensemble: soft vote (average probabilities) if any probability info exists
    proba_sum = {'REAL': 0.0, 'FAKE': 0.0}
    proba_count = 0
    for pmap in prob_maps:
        if not pmap:
            continue
        # ensure both keys exist
        if 'REAL' in pmap and 'FAKE' in pmap:
            proba_sum['REAL'] += pmap['REAL']
            proba_sum['FAKE'] += pmap['FAKE']
            proba_count += 1

    if proba_count > 0:
        avg_proba = {k: (v / proba_count) for k, v in proba_sum.items()}
        ensemb_label = max(avg_proba.items(), key=lambda x: x[1])[0]
        result['ensemble']['soft_vote'] = {'prediction': ensemb_label, 'avg_proba': avg_proba, 'used_models': proba_count}
    else:
        result['ensemble']['soft_vote'] = {'note': 'No probability information available from models; soft voting skipped.'}

    return result

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    form = NewsForm()
    prediction = None
    article_text = None
    user_url = None
    api_check_result = None
    source_type = None

    # Unified GET handling (for links from live feed or direct query)
    if 'url' in request.args or 'text' in request.args:
        if request.args.get('url'):
            user_url = request.args.get('url')
            form.url.data = user_url
            article_text, article_title = get_article_content(user_url)
            source_type = "URL"
        elif request.args.get('text'):
            user_text = request.args.get('text')
            form.text.data = user_text
            article_text = user_text
            article_title = user_text[:50]
            source_type = "TEXT"

        if article_text and source_type:
            # Use new multi-model predictor
            prediction = predict_all(article_text)

            if source_type == "URL":
                api_check_result = check_external_sources(article_title)
            elif source_type == "TEXT":
                api_check_result = "API Check Skipped (Direct text input used)."
        elif source_type:
            prediction = f"⚠️ Error: Could not process content from {source_type}. (Check URL or site security)"
    # Keep POST to satisfy form semantics if used
    elif form.validate_on_submit():
        pass

    return render_template('index.html',
                           form=form,
                           prediction=prediction,
                           article_text=article_text,
                           user_url=user_url,
                           api_check_result=api_check_result)

@app.route('/live_news_feed', methods=['GET', 'POST'])
def live_news_feed():
    if request.method == 'POST':
        selected_url = request.form.get('selected_url')
        if selected_url:
            return redirect(url_for('index', url=selected_url))

    articles = fetch_top_headlines()
    return render_template('live_news.html', articles=articles)

def fetch_top_headlines():
    """Fetches a list of recent headlines using the NewsData.io API."""
    if not NEWS_API_KEY or len(NEWS_API_KEY) < 20:
        return [{"title": "API Key Missing/Invalid", "url": "#", "description": "Please verify NEWS_API_KEY in config.py."}]

    params = {
        'apikey': NEWS_API_KEY,
        'q': 'technology OR finance OR politics',
        'language': 'en',
        'country': 'us,gb,in',
        'size': 10,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        return [{"title": "API Fetch Error", "url": "#", "description": f"Failed to fetch headlines: {e}"}]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
