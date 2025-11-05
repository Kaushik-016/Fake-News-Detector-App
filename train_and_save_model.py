import pandas as pd
import joblib
import os
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed — skipping.")


# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_input_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


# -------------------------------
# LOAD & MERGE DATASETS
# -------------------------------
def load_and_preprocess_data():
    DATA_FILES = [
        {"file": "random_dataset.csv", "label_override": None},
        {"file": "manual_fake_data.csv", "label_override": 0},
        {"file": "manual_real_data.csv", "label_override": 1},
        {"file": "new_real_news_data.csv", "label_override": 1},
    ]

    dfs = []
    # Load your existing datasets
    for src in DATA_FILES:
        if os.path.exists(src["file"]):
            df = pd.read_csv(src["file"], on_bad_lines="skip", encoding="utf-8")
            if "text" not in df.columns:
                continue
            if src["label_override"] is not None:
                df["label"] = src["label_override"]
            dfs.append(df[["text", "label"]])
        else:
            print(f"⚠️ Missing file skipped: {src['file']}")

    # --------------------------
    # Add ISOT Dataset
    # --------------------------
    isot_files = ["Fake.csv", "FakeNews.csv", "True.csv", "TrueNews.csv"]
    isot_found = False

    fake_df, true_df = None, None
    for f in isot_files:
        if os.path.exists(f):
            if "Fake" in f:
                fake_df = pd.read_csv(f, encoding="utf-8")
                fake_df["label"] = 0
                isot_found = True
            elif "True" in f:
                true_df = pd.read_csv(f, encoding="utf-8")
                true_df["label"] = 1
                isot_found = True

    if isot_found and fake_df is not None and true_df is not None:
        if "text" in fake_df.columns:
            f_col = "text"
        elif "title" in fake_df.columns:
            f_col = "title"
        else:
            f_col = fake_df.columns[0]

        if "text" in true_df.columns:
            t_col = "text"
        elif "title" in true_df.columns:
            t_col = "title"
        else:
            t_col = true_df.columns[0]

        isot_df = pd.concat([
            fake_df[[f_col, "label"]].rename(columns={f_col: "text"}),
            true_df[[t_col, "label"]].rename(columns={t_col: "text"})
        ], ignore_index=True)
        dfs.append(isot_df)
        print(f"✅ ISOT dataset loaded successfully with {len(isot_df)} records.")

    if not dfs:
        raise FileNotFoundError("❌ No datasets found!")

    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_input_text)
    df["label"] = df["label"].astype(int)

    print(f"📊 Total combined dataset size: {len(df)} samples")
    print(df["label"].value_counts())

    return df["text"], df["label"]


# -------------------------------
# TRAIN & SAVE MODELS
# -------------------------------
def train_and_save_all_models():
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        sublinear_tf=True,
        ngram_range=(1, 3)
    )
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Balance data
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(tfidf_train, y_train)

    # Models
    classifiers = {
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=120, C=2.0, random_state=42),
        "SVC": SVC(kernel="linear", probability=True, C=2.0, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=0.3),
        "LogisticRegression": LogisticRegression(max_iter=1200, C=2.0, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=70, random_state=42),
    }

    if HAS_XGB:
        classifiers["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.15,
            max_depth=8,
            random_state=42
        )

    print("\n--- Training and Evaluating Models ---")
    results = {}
    best_model, best_acc, best_name = None, 0.0, ""

    for name, clf in classifiers.items():
        clf.fit(X_bal, y_bal)
        y_pred = clf.predict(tfidf_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"[{name}]: Accuracy = {acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, clf, name

    # Calibrate PassiveAggressive
    print("\nCalibrating PassiveAggressive for soft voting ...")
    try:
        calibrated_pa = CalibratedClassifierCV(estimator=classifiers["PassiveAggressive"], cv=3)
    except TypeError:
        calibrated_pa = CalibratedClassifierCV(base_estimator=classifiers["PassiveAggressive"], cv=3)
    calibrated_pa.fit(X_bal, y_bal)

    # Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("pa", calibrated_pa),
            ("svc", classifiers["SVC"]),
            ("rf", classifiers["RandomForest"])
        ],
        voting="soft"
    )
    ensemble.fit(X_bal, y_bal)
    y_pred = ensemble.predict(tfidf_test)
    ensemble_acc = accuracy_score(y_test, y_pred)
    results["Ensemble"] = ensemble_acc
    print(f"[Ensemble]: Accuracy = {ensemble_acc*100:.2f}%")

    if ensemble_acc > best_acc:
        best_acc, best_model, best_name = ensemble_acc, ensemble, "Ensemble"

    print("\n--- Results Summary ---")
    for k, v in results.items():
        print(f" - {k}: {v*100:.2f}%")
    print(f"\n✅ Best Model: {best_name} | Accuracy: {best_acc*100:.2f}%")

    # Retrain on full data
    tfidf_full = tfidf_vectorizer.fit_transform(X)
    joblib.dump(tfidf_vectorizer, "tfidfvect.pkl")

    for name, clf in classifiers.items():
        clf.fit(tfidf_full, y)
        joblib.dump(clf, f"model_{name}.pkl")

    ensemble.fit(tfidf_full, y)
    joblib.dump(ensemble, "model_Ensemble.pkl")
    joblib.dump(best_model, "model.pkl")

    print("\nAll models saved successfully ✅")
    print(f"Highest accuracy achieved: {best_acc*100:.2f}%")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    train_and_save_all_models()
