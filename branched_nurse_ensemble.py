from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score


def nurse_id_from_path(path: Path) -> str:
    return path.stem.replace("processed_nurse_", "")


def infer_window_steps(time_values: np.ndarray, window_seconds: float) -> int:
    if len(time_values) < 2:
        return 1
    diffs = np.diff(time_values)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 1
    dt = float(np.median(positive_diffs))
    return max(1, int(round(window_seconds / dt)))


def build_branched_features(
    X: np.ndarray,
    y: np.ndarray,
    window_steps: int,
    stride_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < window_steps:
        return (
            np.empty((0, X.shape[1] * 6), dtype=np.float32),
            np.empty((0, X.shape[1] * 6), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
        )

    branch_a: list[np.ndarray] = []
    branch_b: list[np.ndarray] = []
    y_windows: list[int] = []

    for start in range(0, len(X) - window_steps + 1, stride_steps):
        end = start + window_steps
        w = X[start:end]

        # Branch A: level/statistics descriptors.
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        minv = w.min(axis=0)
        maxv = w.max(axis=0)
        last = w[-1]
        slope = (w[-1] - w[0]) / max(1, window_steps - 1)
        feat_a = np.concatenate([mean, std, minv, maxv, last, slope], axis=0)

        # Branch B: dynamics/volatility descriptors.
        q25 = np.percentile(w, 25, axis=0)
        median = np.percentile(w, 50, axis=0)
        q75 = np.percentile(w, 75, axis=0)
        iqr = q75 - q25
        diff = np.diff(w, axis=0)
        if len(diff) == 0:
            madiff = np.zeros(w.shape[1], dtype=np.float32)
            stddiff = np.zeros(w.shape[1], dtype=np.float32)
            maxabsdiff = np.zeros(w.shape[1], dtype=np.float32)
        else:
            madiff = np.mean(np.abs(diff), axis=0)
            stddiff = np.std(diff, axis=0)
            maxabsdiff = np.max(np.abs(diff), axis=0)
        feat_b = np.concatenate([median, iqr, madiff, stddiff, maxabsdiff, slope], axis=0)

        branch_a.append(feat_a)
        branch_b.append(feat_b)
        y_windows.append(int(y[end - 1]))

    return (
        np.asarray(branch_a, dtype=np.float32),
        np.asarray(branch_b, dtype=np.float32),
        np.asarray(y_windows, dtype=np.int8),
    )


def load_windows_for_nurse(
    csv_path: Path,
    window_seconds: float,
    fixed_window_steps: int | None,
    stride_fraction: float,
) -> dict[str, Any] | None:
    df = pd.read_csv(csv_path)
    required_cols = {"time", "acc_mag", "EDA", "HR", "TEMP", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.sort_values("time").reset_index(drop=True)
    df["label_bin"] = (df["label"] > 0).astype(int)

    features = ["acc_mag", "EDA", "HR", "TEMP"]
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label_bin"].to_numpy(dtype=np.int8)

    window_steps = (
        fixed_window_steps
        if fixed_window_steps is not None
        else infer_window_steps(df["time"].to_numpy(dtype=np.float32), window_seconds)
    )
    stride_steps = max(1, int(round(window_steps * stride_fraction)))

    Xa, Xb, yw = build_branched_features(X, y, window_steps, stride_steps)
    if len(yw) == 0:
        return None

    return {
        "nurse_id": nurse_id_from_path(csv_path),
        "Xa": Xa,
        "Xb": Xb,
        "yw": yw,
        "window_steps": int(window_steps),
        "stride_steps": int(stride_steps),
        "num_windows": int(len(yw)),
    }


def split_train_calib_test(
    Xa: np.ndarray,
    Xb: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    calib_ratio: float,
) -> dict[str, np.ndarray]:
    if train_ratio + calib_ratio >= 1.0:
        raise ValueError("train_ratio + calib_ratio must be < 1.0")

    n = len(y)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))

    train_end = min(max(train_end, 1), n - 2)
    calib_end = min(max(calib_end, train_end + 1), n - 1)

    return {
        "Xa_train": Xa[:train_end],
        "Xb_train": Xb[:train_end],
        "y_train": y[:train_end],
        "Xa_calib": Xa[train_end:calib_end],
        "Xb_calib": Xb[train_end:calib_end],
        "y_calib": y[train_end:calib_end],
        "Xa_test": Xa[calib_end:],
        "Xb_test": Xb[calib_end:],
        "y_test": y[calib_end:],
    }


def fit_branch_a(X: np.ndarray, y: np.ndarray, random_state: int) -> Any:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"kind": "constant", "class": int(unique_classes[0])}

    neg = int(np.sum(y == 0))
    pos = int(np.sum(y == 1))
    if neg > 0 and pos > 0:
        w0 = len(y) / (2.0 * neg)
        w1 = len(y) / (2.0 * pos)
        sample_weight = np.where(y == 1, w1, w0)
    else:
        sample_weight = None

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        l2_regularization=1e-3,
        random_state=random_state,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_branch_b(X: np.ndarray, y: np.ndarray, random_state: int) -> Any:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"kind": "constant", "class": int(unique_classes[0])}

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)
    return model


def positive_class_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") == "constant":
        cls = int(model["class"])
        return np.ones(len(X), dtype=np.float32) if cls == 1 else np.zeros(len(X), dtype=np.float32)

    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=np.float32)


def find_best_alpha_and_threshold(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y_true: np.ndarray,
    min_recall_0: float,
    min_recall_1: float,
) -> tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0

    best_alpha = 0.5
    best_threshold = 0.5
    best_score = -1.0

    y0 = y_true == 0
    y1 = y_true == 1

    for alpha in np.linspace(0.0, 1.0, 21):
        p = alpha * p_a + (1.0 - alpha) * p_b
        for t in np.linspace(0.10, 0.90, 33):
            y_hat = (p >= t).astype(np.int8)
            recall_0 = float(np.mean(y_hat[y0] == 0)) if np.any(y0) else 0.0
            recall_1 = float(np.mean(y_hat[y1] == 1)) if np.any(y1) else 0.0
            bal_acc = balanced_accuracy_score(y_true, y_hat)
            macro_f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
            base_score = 0.6 * macro_f1 + 0.4 * bal_acc

            # Penalize operating points that violate minimum recall targets.
            miss0 = max(0.0, min_recall_0 - recall_0)
            miss1 = max(0.0, min_recall_1 - recall_1)
            score = base_score - (1.5 * miss0 + 0.8 * miss1)

            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_threshold = float(t)

    return best_alpha, best_threshold, float(best_score)


def eval_and_report(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_bin = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

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
    data_dir = Path("data/Eric")
    window_seconds = 30.0
    window_steps_fixed = None
    stride_fraction = 0.50
    train_ratio = 0.70
    calib_ratio = 0.10
    random_state = 42
    min_recall_0 = 0.25
    min_recall_1 = 0.70

    # Entire held-out nurses for validation.
    val_nurses = ["94", "E4", "6D"]

    model_out = Path("branched_nurse_ensemble.joblib")
    meta_out = Path("branched_nurse_ensemble_meta.json")

    csv_files = sorted(data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No processed_nurse_*.csv found in {data_dir}")

    nurse_windows: dict[str, dict[str, Any]] = {}
    for p in csv_files:
        d = load_windows_for_nurse(p, window_seconds, window_steps_fixed, stride_fraction)
        if d is not None:
            nurse_windows[str(d["nurse_id"])] = d

    all_nurses = sorted(nurse_windows.keys())
    val_nurses = [n for n in val_nurses if n in nurse_windows]
    if len(val_nurses) < 2:
        val_nurses = all_nurses[-3:]

    train_nurses = [n for n in all_nurses if n not in val_nurses]
    if not train_nurses:
        raise RuntimeError("No training nurses remain after validation split.")

    print(f"Train nurses: {train_nurses}")
    print(f"Validation nurses (held out): {val_nurses}")

    splits: dict[str, dict[str, np.ndarray]] = {}
    for n in train_nurses:
        d = nurse_windows[n]
        splits[n] = split_train_calib_test(d["Xa"], d["Xb"], d["yw"], train_ratio, calib_ratio)

    experts: dict[str, dict[str, Any]] = {}
    for i, n in enumerate(train_nurses):
        y_train = splits[n]["y_train"]
        experts[n] = {
            "branch_a": fit_branch_a(splits[n]["Xa_train"], y_train, random_state + i),
            "branch_b": fit_branch_b(splits[n]["Xb_train"], y_train, random_state + 100 + i),
        }

    Xa_calib = np.vstack([splits[n]["Xa_calib"] for n in train_nurses])
    Xb_calib = np.vstack([splits[n]["Xb_calib"] for n in train_nurses])
    y_calib = np.concatenate([splits[n]["y_calib"] for n in train_nurses])

    Xa_test = np.vstack([splits[n]["Xa_test"] for n in train_nurses])
    Xb_test = np.vstack([splits[n]["Xb_test"] for n in train_nurses])
    y_test = np.concatenate([splits[n]["y_test"] for n in train_nurses])

    Xa_val = np.vstack([nurse_windows[n]["Xa"] for n in val_nurses])
    Xb_val = np.vstack([nurse_windows[n]["Xb"] for n in val_nurses])
    y_val = np.concatenate([nurse_windows[n]["yw"] for n in val_nurses])

    # Aggregate expert outputs within each branch, then fuse both branches.
    calib_pa = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_a"], Xa_calib) for n in train_nurses]),
        axis=0,
    )
    calib_pb = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_b"], Xb_calib) for n in train_nurses]),
        axis=0,
    )

    alpha, threshold, calib_score = find_best_alpha_and_threshold(
        calib_pa,
        calib_pb,
        y_calib,
        min_recall_0=min_recall_0,
        min_recall_1=min_recall_1,
    )
    print(f"Calibrated branch weight alpha (A vs B): {alpha:.2f}")
    print(f"Calibrated threshold: {threshold:.2f}")

    test_pa = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_a"], Xa_test) for n in train_nurses]),
        axis=0,
    )
    test_pb = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_b"], Xb_test) for n in train_nurses]),
        axis=0,
    )
    val_pa = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_a"], Xa_val) for n in train_nurses]),
        axis=0,
    )
    val_pb = np.mean(
        np.vstack([positive_class_proba(experts[n]["branch_b"], Xb_val) for n in train_nurses]),
        axis=0,
    )

    test_proba = alpha * test_pa + (1.0 - alpha) * test_pb
    val_proba = alpha * val_pa + (1.0 - alpha) * val_pb
    y_test_pred = (test_proba >= threshold).astype(np.int8)
    y_val_pred = (val_proba >= threshold).astype(np.int8)

    if len(np.unique(y_calib)) > 1:
        y_calib_pred = ((alpha * calib_pa + (1.0 - alpha) * calib_pb) >= threshold).astype(np.int8)
        calib_recall_0 = float(np.mean(y_calib_pred[y_calib == 0] == 0)) if np.any(y_calib == 0) else 0.0
        calib_recall_1 = float(np.mean(y_calib_pred[y_calib == 1] == 1)) if np.any(y_calib == 1) else 0.0
        print(f"Calibration recall_0: {calib_recall_0:.4f}, recall_1: {calib_recall_1:.4f}")
    else:
        calib_recall_0 = 0.0
        calib_recall_1 = 0.0

    test_metrics = eval_and_report("Test (train nurses, unseen time)", y_test, y_test_pred)
    val_metrics = eval_and_report("Validation (held-out nurses)", y_val, y_val_pred)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "experts": experts,
            "alpha": alpha,
            "threshold": threshold,
            "feature_order": ["acc_mag", "EDA", "HR", "TEMP"],
            "branch_a_features": ["mean", "std", "min", "max", "last", "slope"],
            "branch_b_features": ["median", "iqr", "mean_abs_diff", "std_diff", "max_abs_diff", "slope"],
            "val_nurses": val_nurses,
            "train_nurses": train_nurses,
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
        "alpha_branch_a": alpha,
        "alpha_branch_b": 1.0 - alpha,
        "threshold": threshold,
        "calib_objective_score": calib_score,
        "calib_min_recall_0_constraint": min_recall_0,
        "calib_min_recall_1_constraint": min_recall_1,
        "calib_recall_0": calib_recall_0,
        "calib_recall_1": calib_recall_1,
        "test_metrics_train_nurses": test_metrics,
        "validation_metrics_heldout_nurses": val_metrics,
        "label_binarization": "label > 0 -> 1, label == 0 -> 0",
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2))

    print(f"Saved model bundle to: {model_out}")
    print(f"Saved metadata to: {meta_out}")


if __name__ == "__main__":
    main()