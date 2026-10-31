"""Train a metadata-only Indian court disposal baseline from Hugging Face.

It uses court_name, case_type and year only. It deliberately excludes judgment
full_text because final judgments disclose their own outcome (target leakage).
Run: python train_outcome_model.py --max-train 150000 --max-eval 20000
"""
import argparse, json
from pathlib import Path
from datasets import load_dataset
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET = "overthelex/indian-court-decisions"
CONFIG = "high_courts"
FEATURES = ["court_name", "case_type", "year"]
LABEL = "disposal_nature_normalized"


def collect(split, limit):
    print(f"Streaming {split}: up to {limit:,} usable rows")
    stream = load_dataset(DATASET, CONFIG, split=split, streaming=True)
    rows = []
    for r in stream:
        label = str(r.get(LABEL) or "").strip().lower()
        court = str(r.get("court_name") or "").strip()
        case_type = str(r.get("case_type") or "").strip()
        try:
            year = int(r.get("year"))
        except (TypeError, ValueError):
            continue
        if not label or label in {"none", "null", "nan"} or not court or not case_type or not 1950 <= year <= 2035:
            continue
        rows.append({"court_name": court, "case_type": case_type, "year": year, "outcome": label})
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No valid rows in {split}. Check dataset ID/config and internet access.")
    print(f"Collected {len(rows):,}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=150000)
    ap.add_argument("--max-eval", type=int, default=20000)
    args = ap.parse_args()
    train = collect("train", args.max_train)
    valid = collect("validation", args.max_eval)
    test = collect("test", args.max_eval)
    supported = set(train.outcome.value_counts().loc[lambda x: x >= 100].index)
    train, valid, test = [df[df.outcome.isin(supported)].copy() for df in (train, valid, test)]
    if train.outcome.nunique() < 2 or valid.empty or test.empty:
        raise RuntimeError("Insufficient supported classes for train/validation/test.")
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["court_name", "case_type"]),
        ("year", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["year"]),
    ])
    model = Pipeline([("features", prep), ("classifier", LogisticRegression(
        max_iter=1200, class_weight="balanced", solver="saga", n_jobs=-1, random_state=42
    ))])
    model.fit(train[FEATURES], train.outcome)
    valid_pred = model.predict(valid[FEATURES])
    test_pred = model.predict(test[FEATURES])
    metrics = {
        "dataset": DATASET, "configuration": CONFIG, "features": FEATURES, "label": LABEL,
        "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
        "classes": sorted(supported),
        "validation_macro_f1": float(f1_score(valid.outcome, valid_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(test.outcome, test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test.outcome, test_pred)),
        "test_macro_f1": float(f1_score(test.outcome, test_pred, average="macro", zero_division=0)),
        "test_classification_report": classification_report(test.outcome, test_pred, output_dict=True, zero_division=0),
        "interpretation": "Metadata-only historical disposal category baseline; not a merits or individual-case outcome predictor."
    }
    joblib.dump(model, ROOT / "outcome_model.joblib")
    (ROOT / "outcome_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("train_rows", "test_rows", "test_accuracy", "test_balanced_accuracy", "test_macro_f1")}, indent=2))
    print("Saved outcome_model.joblib and outcome_model_metrics.json beside app.py")

if __name__ == "__main__":
    main()
