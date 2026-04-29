"""Global XGBoost baseline for the nurse stress dataset.

The script trains one XGBoost model per training nurse, then combines their
probabilities with calibration-based weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier


def nurse_id_from_path(path: Path) -> str:
    """Convert a processed nurse filename into the short nurse identifier."""
    return path.stem.replace("processed_nurse_", "")


def infer_window_steps(time_values: np.ndarray, window_seconds: float) -> int:
    """Estimate the number of samples that correspond to one window."""
    if len(time_values) < 2:
        return 1
    diffs = np.diff(time_values)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 1
    dt = float(np.median(positive_diffs))
    return max(1, int(round(window_seconds / dt)))


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_steps: int,
    stride_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize each sliding window with simple statistics and a slope."""
    if len(X) < window_steps:
        return np.empty((0, X.shape[1] * 6), dtype=np.float32), np.empty((0,), dtype=np.int8)

    # Each window is represented by a compact feature vector rather than raw samples.
    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []

    for start in range(0, len(X) - window_steps + 1, stride_steps):
        end = start + window_steps
        w = X[start:end]
        # Capture the local level, spread, extrema, endpoint, and trend.
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        minv = w.min(axis=0)
        maxv = w.max(axis=0)
        last = w[-1]
        slope = (w[-1] - w[0]) / max(1, window_steps - 1)
        X_windows.append(np.concatenate([mean, std, minv, maxv, last, slope], axis=0))
        y_windows.append(int(y[end - 1]))

    return np.asarray(X_windows, dtype=np.float32), np.asarray(y_windows, dtype=np.int8)


def load_windows_for_nurse(
    csv_path: Path,
    window_seconds: float,
    fixed_window_steps: int | None,
    stride_fraction: float,
) -> dict[str, Any] | None:
    """Convert one nurse CSV into a windowed train/calibration/test bundle."""
    df = pd.read_csv(csv_path)
    # The model only uses the raw sensor channels plus the binary stress label.
    required_cols = {"time", "acc_mag", "EDA", "HR", "TEMP", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    # Sort by time so all later splits preserve the original temporal order.
    df = df.sort_values("time").reset_index(drop=True)
    df["label_bin"] = (df["label"] > 0).astype(int)

    # Convert the raw dataframe into numpy arrays for the window builder.
    features = ["acc_mag", "EDA", "HR", "TEMP"]
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label_bin"].to_numpy(dtype=np.int8)

    # Derive a window size from the observed sample spacing when needed.
    window_steps = (
        fixed_window_steps
        if fixed_window_steps is not None
        else infer_window_steps(df["time"].to_numpy(dtype=np.float32), window_seconds)
    )
    stride_steps = max(1, int(round(window_steps * stride_fraction)))

    # Build the rolling windows once; the rest of the pipeline reuses them.
    Xw, yw = build_windows(X, y, window_steps, stride_steps)
    if len(Xw) == 0:
        return None

    return {
        "nurse_id": nurse_id_from_path(csv_path),
        "Xw": Xw,
        "yw": yw,
        "window_steps": int(window_steps),
        "stride_steps": int(stride_steps),
        "num_windows": int(len(Xw)),
    }


def split_train_calib_test(
    Xw: np.ndarray,
    yw: np.ndarray,
    train_ratio: float,
    calib_ratio: float,
) -> dict[str, np.ndarray]:
    """Split the windowed data into contiguous train, calibration, and test chunks."""
    if train_ratio + calib_ratio >= 1.0:
        raise ValueError("train_ratio + calib_ratio must be < 1.0")

    # Keep the split contiguous so the evaluation respects time order.
    n = len(Xw)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))

    train_end = min(max(train_end, 1), n - 2)
    calib_end = min(max(calib_end, train_end + 1), n - 1)

    return {
        "X_train": Xw[:train_end],
        "y_train": yw[:train_end],
        "X_calib": Xw[train_end:calib_end],
        "y_calib": yw[train_end:calib_end],
        "X_test": Xw[calib_end:],
        "y_test": yw[calib_end:],
    }


def build_xgb(random_state: int) -> XGBClassifier:
    """Construct the XGBoost model used for each nurse-specific learner."""
    # These hyperparameters match the rest of the repository's tuned baseline.
    return XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def positive_class_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return the positive-class probability, even for constant fallback models."""
    if isinstance(model, dict) and model.get("kind") == "constant":
        # Single-class nurses are represented by a constant predictor.
        cls = int(model["class"])
        return np.ones(len(X), dtype=np.float32) if cls == 1 else np.zeros(len(X), dtype=np.float32)

    # For normal classifiers, select the column corresponding to class 1.
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=np.float32)


def find_threshold(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Select the threshold that maximizes the calibration objective."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    # Sweep a coarse grid of candidate cutoffs instead of overfitting to a single value.
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.10, 0.90, 33):
        y_hat = (proba >= t).astype(np.int8)
        bal_acc = balanced_accuracy_score(y_true, y_hat)
        macro_f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
        # Blend macro-F1 and balanced accuracy so neither class dominates the choice.
        score = 0.6 * macro_f1 + 0.4 * bal_acc
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, float(best_score)


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Print and return the main metrics for one evaluation split."""
    # Keep the metric calculation explicit so the printed summary matches the JSON output.
    acc = accuracy_score(y_true, y_pred)
    f1_bin = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if len(np.unique(y_true)) > 1:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    else:
        bal_acc = 0.0

    print(f"=== {split_name} Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (binary): {f1_bin:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": float(acc),
        "f1_binary": float(f1_bin),
        "f1_macro": float(f1_macro),
        "balanced_accuracy": float(bal_acc),
    }


def main() -> None:
    """Train the per-nurse XGBoost ensemble and evaluate it globally."""

    data_dir = Path("data/Eric")
    window_seconds = 30.0
    window_steps_fixed = None
    stride_fraction = 0.50
    train_ratio = 0.7
    calib_ratio = 0.1
    random_state = 42

    # Entire nurses are held out for validation, so the ensemble is tested on unseen people.
    val_nurses = ["94", "E4", "6D"]

    model_out = Path("xgb_nurse_ensemble.joblib")
    meta_out = Path("xgb_nurse_ensemble_meta.json")

    # Load every nurse once, then reuse the cached windows for training and validation.
    csv_files = sorted(data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No processed_nurse_*.csv found in {data_dir}")

    # Cache each nurse's windowed dataset once so the later ensemble steps are cheap.
    nurse_windows: dict[str, dict[str, Any]] = {}
    for p in csv_files:
        loaded = load_windows_for_nurse(p, window_seconds, window_steps_fixed, stride_fraction)
        if loaded is not None:
            nurse_windows[str(loaded["nurse_id"])] = loaded

    if not nurse_windows:
        raise RuntimeError("No nurse had enough samples for configured window/stride.")

    # Hold out entire nurses first, then fall back to the last three if needed.
    all_nurses = sorted(nurse_windows.keys())
    val_nurses = [n for n in val_nurses if n in nurse_windows]
    if len(val_nurses) < 2:
        val_nurses = all_nurses[-3:]

    # The remaining nurses form the training pool for the per-nurse ensemble.
    train_nurses = [n for n in all_nurses if n not in val_nurses]
    if not train_nurses:
        raise RuntimeError("No training nurses remain after validation selection.")

    print(f"Train nurses: {train_nurses}")
    print(f"Validation nurses (held out): {val_nurses}")

    # Split each training nurse into train/calibration/test chunks.
    train_splits: dict[str, dict[str, np.ndarray]] = {}
    for nurse_id in train_nurses:
        d = nurse_windows[nurse_id]
        split = split_train_calib_test(d["Xw"], d["yw"], train_ratio, calib_ratio)
        train_splits[nurse_id] = split

    # Fit one model per training nurse rather than pooling all nurses together.
    models: dict[str, Any] = {}
    for i, nurse_id in enumerate(train_nurses):
        X_train = train_splits[nurse_id]["X_train"]
        y_train = train_splits[nurse_id]["y_train"]
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # Preserve nurses that only contain one class in the training slice.
            models[nurse_id] = {"kind": "constant", "class": int(unique_classes[0])}
            continue
        m = build_xgb(random_state + i)
        m.fit(X_train, y_train)
        models[nurse_id] = m

    X_calib = np.vstack([train_splits[n]["X_calib"] for n in train_nurses])
    y_calib = np.concatenate([train_splits[n]["y_calib"] for n in train_nurses])

    X_test_train = np.vstack([train_splits[n]["X_test"] for n in train_nurses])
    y_test_train = np.concatenate([train_splits[n]["y_test"] for n in train_nurses])

    X_val = np.vstack([nurse_windows[n]["Xw"] for n in val_nurses])
    y_val = np.concatenate([nurse_windows[n]["yw"] for n in val_nurses])

    # Stack probabilities so we can average nurse models with shared weights.
    model_ids = list(models.keys())
    calib_stack = np.vstack([positive_class_proba(models[n], X_calib) for n in model_ids])
    test_stack = np.vstack([positive_class_proba(models[n], X_test_train) for n in model_ids])
    val_stack = np.vstack([positive_class_proba(models[n], X_val) for n in model_ids])

    # Weight models by calibration balanced accuracy.
    raw_weights = []
    model_weights: dict[str, float] = {}
    for i, nurse_id in enumerate(model_ids):
        # Convert calibration probabilities into a quick hard prediction for weighting.
        preds = (calib_stack[i] >= 0.5).astype(np.int8)
        if len(np.unique(y_calib)) > 1:
            w = max(0.05, float(balanced_accuracy_score(y_calib, preds)))
        else:
            w = 1.0
        raw_weights.append(w)
        model_weights[nurse_id] = w

    weights = np.asarray(raw_weights, dtype=np.float32)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    # Tune one final threshold on the calibration ensemble before scoring the held-out sets.
    calib_ensemble_proba = np.average(calib_stack, axis=0, weights=weights)
    threshold, calib_score = find_threshold(calib_ensemble_proba, y_calib)
    print(f"Calibrated threshold: {threshold:.2f}")

    # Use the same ensemble weights on both the within-training test split and the held-out nurses.
    y_pred_test_train = (np.average(test_stack, axis=0, weights=weights) >= threshold).astype(np.int8)
    y_pred_val = (np.average(val_stack, axis=0, weights=weights) >= threshold).astype(np.int8)

    test_metrics = evaluate_split("Test (train nurses, unseen time)", y_test_train, y_pred_test_train)
    val_metrics = evaluate_split("Validation (held-out nurses)", y_val, y_pred_val)

    # Save the trained ensemble and the metadata needed to reproduce the run.
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "model_weights": model_weights,
            "normalized_weights": {model_ids[i]: float(weights[i]) for i in range(len(model_ids))},
            "threshold": threshold,
            "feature_order": ["acc_mag", "EDA", "HR", "TEMP"],
            "window_features": ["mean", "std", "min", "max", "last", "slope"],
        },
        model_out,
    )

    meta = {
        "data_dir": str(data_dir),
        "window_seconds": window_seconds,
        "window_steps_fixed": window_steps_fixed,
        "stride_fraction": stride_fraction,
        "train_ratio": train_ratio,
        "calib_ratio": calib_ratio,
        "test_ratio": 1.0 - train_ratio - calib_ratio,
        "random_state": random_state,
        "val_nurses": val_nurses,
        "train_nurses": train_nurses,
        "threshold": threshold,
        "calib_objective_score": calib_score,
        "test_metrics_train_nurses": test_metrics,
        "validation_metrics_heldout_nurses": val_metrics,
        "model_weights": model_weights,
        "nurse_window_config": {
            n: {
                "window_steps": int(nurse_windows[n]["window_steps"]),
                "stride_steps": int(nurse_windows[n]["stride_steps"]),
                "num_windows": int(nurse_windows[n]["num_windows"]),
            }
            for n in all_nurses
        },
        "label_binarization": "label > 0 -> 1, label == 0 -> 0",
    }

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2))

    print(f"Saved model bundle to: {model_out}")
    print(f"Saved metadata to: {meta_out}")


if __name__ == "__main__":
    main()