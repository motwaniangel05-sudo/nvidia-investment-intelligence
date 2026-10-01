"""
Train a classical ML text classifier (TF-IDF + LogisticRegression) to
label sentences as FACT / CLAIM / INTERPRETATION / UNCERTAINTY.

Per project constraints: no pretrained LLMs, classical/transparent ML
only. Trained entirely on the 85 manually-labeled real sentences from
Phase 10 (data/processed/labeling_template.csv).

HONEST LIMITATION: 85 examples across 4 classes (one class, CLAIM, has
only 5) is a small training set. Cross-validation is used instead of a
single train/test split to get a more reliable estimate from limited
data, but absolute performance should be interpreted cautiously --
this is a proof-of-concept classifier, not a production-grade one.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core.config_loader import get_path, load_config
from core.logger import get_logger

log = get_logger(__name__)

LABELS = ["FACT", "CLAIM", "INTERPRETATION", "UNCERTAINTY"]


def load_labeled_data() -> pd.DataFrame:
    """Load manually labeled sentences, excluding SKIP rows (fragments)."""
    config = load_config()
    path = get_path(config, "data_processed") / "labeling_template.csv"
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found at {path}. Run Phase 10 labeling first.")

    df = pd.read_csv(path)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Train TF-IDF + LogisticRegression using scikit-learn's built-in
    TfidfVectorizer here (not our Phase 5 from-scratch version) -- this
    is a deliberate choice: Phase 5 proved we understand TF-IDF math by
    building it ourselves; here we use the trusted library implementation
    since the focus of this phase is classification methodology, not
    re-proving vectorization. Evaluated via stratified k-fold cross-
    validation given the small dataset size.
    """
    # .to_numpy(dtype=object) avoids a pandas/pyarrow ChunkedArray vs
    # scikit-learn fancy-indexing incompatibility seen with newer pandas
    # versions that back string columns with PyArrow by default.
    X_text = df["sentence"].to_numpy(dtype=object)
    y = df["label"].to_numpy(dtype=object)

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", min_df=1, max_features=2000
    )
    X = vectorizer.fit_transform(X_text)

    # min class count determines max usable folds (StratifiedKFold needs
    # at least n_folds examples in the smallest class)
    min_class_count = pd.Series(y).value_counts().min()
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        raise ValueError(
            f"Smallest class has only {min_class_count} example(s); "
            f"need at least 2 for cross-validation."
        )
    if actual_folds < n_folds:
        log.warning(
            "Reducing folds from %d to %d: smallest class (likely CLAIM) "
            "has only %d examples.", n_folds, actual_folds, min_class_count,
        )

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    y_pred = cross_val_predict(model, X, y, cv=skf)

    report = classification_report(y, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(y, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=LABELS)

    # Train the FINAL model on all data (cross-validation was only for
    # honest evaluation; the deployed model uses every labeled example)
    model.fit(X, y)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "report_dict": report,
        "report_text": report_text,
        "confusion_matrix": cm,
        "n_folds_used": actual_folds,
        "n_examples": len(df),
        "class_counts": pd.Series(y).value_counts().to_dict(),
    }


def save_classifier(model, vectorizer, path: Path = None) -> Path:
    config = load_config()
    if path is None:
        path = get_path(config, "models") / "classification" / "event_classifier.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "labels": LABELS}, f)
    log.info("Saved classifier to %s", path)
    return path


if __name__ == "__main__":
    df = load_labeled_data()
    print(f"Loaded {len(df)} usable labeled examples (SKIP rows excluded)")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    results = train_and_evaluate(df)

    print(f"Cross-validation folds used: {results['n_folds_used']}")
    print(f"\n--- Classification Report (cross-validated) ---")
    print(results["report_text"])

    print(f"--- Confusion Matrix ---")
    print(f"Rows=actual, Cols=predicted, Order={LABELS}")
    print(results["confusion_matrix"])

    save_path = save_classifier(results["model"], results["vectorizer"])
    print(f"\nFinal model (trained on all {results['n_examples']} examples) saved to: {save_path}")
