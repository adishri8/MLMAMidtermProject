from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


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


def build_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_steps: int,
    stride_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < window_steps:
        return np.empty((0, X.shape[1] * 6)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    end_indices: list[int] = []

    for start in range(0, len(X) - window_steps + 1, stride_steps):
        end = start + window_steps
        w = X[start:end]
        # Summarize local behavior over the window to keep dimensions compact.
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        minv = w.min(axis=0)
        maxv = w.max(axis=0)
        last = w[-1]
        slope = (w[-1] - w[0]) / max(1, window_steps - 1)
        X_windows.append(np.concatenate([mean, std, minv, maxv, last, slope], axis=0))
        # Predict stress at the current endpoint from recent history.
        y_windows.append(int(y[end - 1]))
        end_indices.append(end - 1)

    return np.asarray(X_windows), np.asarray(y_windows), np.asarray(end_indices)


def load_and_process_nurse(
    csv_path: Path,
    window_seconds: float,
    fixed_window_steps: int | None,
    stride_fraction: float,
    test_ratio: float,
) -> dict[str, object] | None:
    df = pd.read_csv(csv_path)

    required_cols = {"time", "acc_mag", "EDA", "HR", "TEMP", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

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

    Xw, yw, end_idx = build_windows(X, y, window_steps, stride_steps)
    if len(Xw) == 0:
        return None

    split_idx = int(len(Xw) * (1.0 - test_ratio))
    split_idx = min(max(split_idx, 1), len(Xw) - 1)

    nurse_id = nurse_id_from_path(csv_path)
    return {
        "nurse_id": nurse_id,
        "window_steps": int(window_steps),
        "stride_steps": int(stride_steps),
        "num_windows": int(len(Xw)),
        "X_train": Xw[:split_idx],
        "y_train": yw[:split_idx],
        "X_test": Xw[split_idx:],
        "y_test": yw[split_idx:],
        "test_end_idx": end_idx[split_idx:],
        "test_times": df.loc[end_idx[split_idx:], "time"].to_numpy(dtype=np.float32),
    }


def main() -> None:
    # Configuration
    data_dir = Path("data/Eric")
    window_seconds = 30.0
    window_steps_fixed = None
    stride_fraction = 0.50
    test_ratio = 0.2
    n_estimators = 300
    max_depth = None
    random_state = 42
    model_out = Path("rf_eric_sliding_window.joblib")
    meta_out = Path("rf_eric_sliding_window_meta.json")

    csv_files = sorted(data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No processed_nurse_*.csv files found in {args.data_dir}")

    nurse_data: list[dict[str, object]] = []
    for csv_path in csv_files:
        result = load_and_process_nurse(
            csv_path=csv_path,
            window_seconds=window_seconds,
            fixed_window_steps=window_steps_fixed,
            stride_fraction=stride_fraction,
            test_ratio=test_ratio,
        )
        if result is not None:
            nurse_data.append(result)

    if not nurse_data:
        raise RuntimeError("No nurse had enough samples for the selected window/stride settings.")

    X_train = np.vstack([d["X_train"] for d in nurse_data])
    y_train = np.concatenate([d["y_train"] for d in nurse_data])
    X_test = np.vstack([d["X_test"] for d in nurse_data])
    y_test = np.concatenate([d["y_test"] for d in nurse_data])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    global_accuracy = accuracy_score(y_test, y_pred)
    global_f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    print("=== Global Test Metrics ===")
    print(f"Accuracy: {global_accuracy:.4f}")
    print(f"F1 (binary): {global_f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    per_nurse_metrics: dict[str, dict[str, float | int]] = {}
    cursor = 0
    print("=== Per-Nurse Test Metrics ===")
    for d in nurse_data:
        nurse_id = str(d["nurse_id"])
        n = len(d["X_test"])
        nurse_pred = y_pred[cursor : cursor + n]
        nurse_true = d["y_test"]
        cursor += n

        acc = accuracy_score(nurse_true, nurse_pred)
        f1 = f1_score(nurse_true, nurse_pred, average="binary", zero_division=0)
        per_nurse_metrics[nurse_id] = {
            "num_test_windows": int(n),
            "accuracy": float(acc),
            "f1_binary": float(f1),
            "positive_rate_true": float(np.mean(nurse_true)),
            "positive_rate_pred": float(np.mean(nurse_pred)),
        }
        print(
            f"Nurse {nurse_id}: windows={n}, accuracy={acc:.4f}, f1={f1:.4f}, "
            f"true_pos_rate={np.mean(nurse_true):.4f}"
        )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)

    metadata = {
        "data_dir": str(data_dir),
        "window_seconds": window_seconds,
        "window_steps_fixed": window_steps_fixed,
        "stride_fraction": stride_fraction,
        "test_ratio": test_ratio,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "num_train_windows": int(len(X_train)),
        "num_test_windows": int(len(X_test)),
        "global_accuracy": float(global_accuracy),
        "global_f1_binary": float(global_f1),
        "nurse_window_config": {
            str(d["nurse_id"]): {
                "window_steps": int(d["window_steps"]),
                "stride_steps": int(d["stride_steps"]),
                "num_windows": int(d["num_windows"]),
            }
            for d in nurse_data
        },
        "per_nurse_test_metrics": per_nurse_metrics,
        "features_per_step": ["acc_mag", "EDA", "HR", "TEMP"],
        "window_label_definition": "label at the final sample of each window",
        "label_binarization": "label > 0 -> 1, label == 0 -> 0",
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model to: {model_out}")
    print(f"Saved metadata to: {meta_out}")


if __name__ == "__main__":
    main()