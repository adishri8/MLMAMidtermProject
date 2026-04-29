"""
Global sliding-window random forest classifier.
- Global z-score normalization across all nurses
- Sliding windows with aggregated statistics
- Leave-one-nurse-day-out validation (LONO-day)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


RAW_FEATURES = ["acc_mag", "EDA", "HR", "TEMP"]


def nurse_id_from_path(path: Path) -> str:
    """Extract nurse ID from CSV filename."""
    return path.stem.replace("processed_nurse_", "")


def load_nurse_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate nurse CSV, compute time features."""
    df = pd.read_csv(csv_path)
    required = {"datetime", "time", "acc_mag", "EDA", "HR", "TEMP", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["label_stressed"] = (df["label"] > 0).astype(np.int8)
    df["day"] = df["datetime"].dt.strftime("%Y-%m-%d")
    day_start = df["datetime"].dt.floor("D")
    df["day_time_sec"] = (df["datetime"] - day_start).dt.total_seconds().astype(np.float32)
    return df


def set_binary_target(df: pd.DataFrame, positive_class: str) -> pd.DataFrame:
    """Set label_bin for requested positive class."""
    df = df.copy()
    stressed = df["label_stressed"].to_numpy(dtype=np.int8)
    if positive_class == "stressed":
        df["label_bin"] = stressed
    elif positive_class == "unstressed":
        df["label_bin"] = (1 - stressed).astype(np.int8)
    else:
        raise ValueError(f"Unknown positive_class: {positive_class}")
    return df


def infer_step_seconds(time_values: np.ndarray) -> float:
    """Estimate the sampling interval from successive timestamps."""
    """Infer sampling interval from time values."""
    if len(time_values) < 2:
        return 1.0
    diffs = np.diff(time_values)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))


def compute_global_normalization(df_source: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute the global mean and standard deviation for each raw sensor."""
    """Compute global z-score normalization parameters across all nurses."""
    means = df_source[RAW_FEATURES].mean(axis=0)
    stds = df_source[RAW_FEATURES].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def apply_z_normalization(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Attach z-scored copies of the raw sensor channels."""
    """Apply z-score normalization to raw features."""
    df = df.copy()
    for col in RAW_FEATURES:
        df[f"{col}_z"] = (df[col] - float(means[col])) / float(stds[col])
    return df


def build_sliding_windows(
    df: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one nurse-day dataframe into summary-statistic windows."""
    """
    Build sliding windows from normalized features.
    
    Returns:
        X_windows: (n_windows, n_features) - aggregated window features
        y_windows: (n_windows,) - label at window end
    """
    feat_cols = [f"{c}_z" for c in RAW_FEATURES]
    X = df[feat_cols].to_numpy(dtype=np.float32)  # (n_samples, 4)
    y = df["label_bin"].to_numpy(dtype=np.int8)
    t = df["day_time_sec"].to_numpy(dtype=np.float32)

    step = infer_step_seconds(t)
    window_steps = max(1, int(round(window_seconds / step)))
    stride_steps = max(1, int(round(stride_seconds / step)))

    if len(X) < window_steps:
        return np.empty((0, len(RAW_FEATURES) * 6), dtype=np.float32), np.empty((0,), dtype=np.int8)

    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []

    for start in range(0, len(X) - window_steps + 1, stride_steps):
        end = start + window_steps
        w = X[start:end]  # window of normalized features

        # Compute aggregated statistics over the window
        mean = w.mean(axis=0)      # (4,)
        std = w.std(axis=0)        # (4,)
        minv = w.min(axis=0)       # (4,)
        maxv = w.max(axis=0)       # (4,)
        last = w[-1]               # (4,)
        slope = (w[-1] - w[0]) / max(1, window_steps - 1)  # (4,)

        # Concatenate all features: 6 aggregations * 4 features = 24 features total
        features = np.concatenate([mean, std, minv, maxv, last, slope], axis=0)
        X_windows.append(features)
        y_windows.append(int(y[end - 1]))

    return np.asarray(X_windows, dtype=np.float32), np.asarray(y_windows, dtype=np.int8)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[RandomForestClassifier | None, np.ndarray, np.ndarray]:
    """Train a random forest and return predictions for the eval split."""
    if len(np.unique(y_train)) < 2:
        # Single class - return constant predictions
        constant = int(y_train[0]) if len(y_train) > 0 else 1
        pred = np.full(shape=(len(X_eval),), fill_value=constant, dtype=np.int8)
        return None, pred, pred

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= 0.5).astype(np.int8)
    return model, y_pred, y_proba


def balance_train_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mode: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally undersample the majority class in the training set."""
    if mode == "none":
        return X_train, y_train

    if mode != "undersample":
        raise ValueError(f"Unknown balance mode: {mode}")

    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X_train, y_train

    n = min(len(idx0), len(idx1))
    rng = np.random.default_rng(random_state)
    keep0 = rng.choice(idx0, size=n, replace=False)
    keep1 = rng.choice(idx1, size=n, replace=False)
    keep = np.concatenate([keep0, keep1])
    rng.shuffle(keep)
    return X_train[keep], y_train[keep]


def find_optimal_threshold_balanced_acc(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes balanced accuracy."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_bal_acc = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_proba >= t).astype(np.int8)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(t)
    return best_threshold, best_bal_acc


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common metrics without failing on edge cases."""
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_samples": 0,
        }

    bal_acc = 0.0
    if len(np.unique(y_true)) > 1:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": bal_acc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
    }


def has_min_class_mix(
    y: np.ndarray,
    min_total: int,
    min_c0: int,
    min_c1: int,
    min_c0_rate: float,
    min_c1_rate: float,
) -> bool:
    n = len(y)
    if n < min_total:
        return False

    c0 = int(np.sum(y == 0))
    c1 = int(np.sum(y == 1))
    if c0 < min_c0 or c1 < min_c1:
        return False

    r0 = c0 / n
    r1 = c1 / n
    return r0 >= min_c0_rate and r1 >= min_c1_rate


def collect_global_data(data_dir: Path, discard_nurses: set[str]) -> pd.DataFrame:
    """Load all nurse CSV files except the discarded IDs."""
    rows: list[pd.DataFrame] = []
    csv_files = sorted(data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No nurse CSV files found in {data_dir}")

    for csv_path in csv_files:
        nurse_id = nurse_id_from_path(csv_path)
        if nurse_id in discard_nurses:
            continue
        df = load_nurse_csv(csv_path)
        df["nurse_id"] = nurse_id
        rows.append(df)

    if not rows:
        raise RuntimeError("No nurse data left after discard filter.")
    return pd.concat(rows, axis=0, ignore_index=True)


def build_windows_by_nurse_day(
    df_norm: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    windows_by_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grouped = df_norm.groupby(["nurse_id", "day"], sort=True)

    for (nurse_id, day), g in grouped:
        Xw, yw = build_sliding_windows(g, window_seconds=window_seconds, stride_seconds=stride_seconds)
        if len(yw) == 0:
            continue
        group_key = f"{nurse_id}|{day}"
        windows_by_group[group_key] = (Xw, yw)

    return windows_by_group


def run_global_lono_day(
    args: argparse.Namespace,
    positive_class: str,
    out_dir: Path,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    discard = set(args.discard_nurses)

    print(f"Loading all nurses and computing global normalization for target={positive_class}...")
    df_all = collect_global_data(args.data_dir, discard)
    df_all = set_binary_target(df_all, positive_class=positive_class)
    means, stds = compute_global_normalization(df_all)
    df_norm = apply_z_normalization(df_all, means, stds)

    print("Building sliding windows per nurse-day...")
    windows_by_group = build_windows_by_nurse_day(
        df_norm,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    group_keys = sorted(windows_by_group.keys())
    if len(group_keys) < 2:
        raise RuntimeError("Need at least 2 nurse-day groups with windows for validation.")

    fold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for fold_id, test_key in enumerate(group_keys, start=1):
        train_keys = [k for k in group_keys if k != test_key]
        X_train_full = np.vstack([windows_by_group[k][0] for k in train_keys])
        y_train_full = np.concatenate([windows_by_group[k][1] for k in train_keys])
        X_test, y_test = windows_by_group[test_key]

        train_ok = has_min_class_mix(
            y=y_train_full,
            min_total=args.min_train_windows,
            min_c0=args.min_train_c0,
            min_c1=args.min_train_c1,
            min_c0_rate=args.min_train_c0_rate,
            min_c1_rate=args.min_train_c1_rate,
        )
        test_ok = has_min_class_mix(
            y=y_test,
            min_total=args.min_test_windows,
            min_c0=args.min_test_c0,
            min_c1=args.min_test_c1,
            min_c0_rate=args.min_test_c0_rate,
            min_c1_rate=args.min_test_c1_rate,
        )
        if not (train_ok and test_ok):
            continue

        train_pos_rate = float(np.mean(y_train_full))
        test_pos_rate = float(np.mean(y_test))
        class_ratio_diff = abs(train_pos_rate - test_pos_rate)
        if class_ratio_diff > args.max_class_ratio_diff:
            continue

        X_train, y_train = balance_train_data(
            X_train_full,
            y_train_full,
            mode=args.balance_train,
            random_state=args.random_state + fold_id,
        )

        model, _, y_proba_train = train_random_forest(X_train, y_train, X_train)
        if args.threshold_strategy == "balanced_acc":
            threshold, threshold_train_bal_acc = find_optimal_threshold_balanced_acc(y_train, y_proba_train)
        else:
            threshold = 0.5
            threshold_train_bal_acc = float(balanced_accuracy_score(y_train, (y_proba_train >= 0.5).astype(np.int8))) if len(np.unique(y_train)) > 1 else 0.0

        if model is None:
            y_pred_test = np.full(shape=(len(X_test),), fill_value=int(y_train[0]) if len(y_train) > 0 else 1, dtype=np.int8)
            y_pred_train = np.full(shape=(len(X_train),), fill_value=int(y_train[0]) if len(y_train) > 0 else 1, dtype=np.int8)
        else:
            y_proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_proba_test >= threshold).astype(np.int8)
            y_pred_train = (y_proba_train >= threshold).astype(np.int8)

        train_metrics = safe_metrics(y_train, y_pred_train)
        test_metrics = safe_metrics(y_test, y_pred_test)

        nurse_id, day = test_key.split("|", 1)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "test_nurse_id": nurse_id,
                "test_day": day,
                "train_group_count": len(train_keys),
                "train_windows_before_balance": int(len(y_train_full)),
                "train_windows_after_balance": int(len(y_train)),
                "test_windows": int(len(y_test)),
                "train_c0": int(np.sum(y_train == 0)),
                "train_c1": int(np.sum(y_train == 1)),
                "test_c0": int(np.sum(y_test == 0)),
                "test_c1": int(np.sum(y_test == 1)),
                "train_pos_rate": float(np.mean(y_train)),
                "test_pos_rate": test_pos_rate,
                "class_ratio_diff": class_ratio_diff,
                "threshold": threshold,
                "threshold_train_balanced_accuracy": threshold_train_bal_acc,
            }
        )

        metric_rows.append(
            {
                "fold_id": fold_id,
                "test_nurse_id": nurse_id,
                "test_day": day,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_f1": train_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "n_test_samples": test_metrics["n_samples"],
                "threshold": threshold,
            }
        )

    if not metric_rows:
        raise RuntimeError("No valid LONO-day folds remained after filtering. Relax thresholds.")

    fold_df = pd.DataFrame(fold_rows).sort_values(["test_nurse_id", "test_day"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["test_nurse_id", "test_day"]).reset_index(drop=True)

    # Train one final global model on all available windows.
    X_all = np.vstack([windows_by_group[k][0] for k in group_keys])
    y_all = np.concatenate([windows_by_group[k][1] for k in group_keys])
    X_all_bal, y_all_bal = balance_train_data(
        X_all,
        y_all,
        mode=args.balance_train,
        random_state=args.random_state,
    )
    final_model, _, y_proba_all = train_random_forest(X_all_bal, y_all_bal, X_all_bal)
    if args.threshold_strategy == "balanced_acc":
        final_threshold, _ = find_optimal_threshold_balanced_acc(y_all_bal, y_proba_all)
    else:
        final_threshold = 0.5
    final_train_metrics = safe_metrics(y_all_bal, (y_proba_all >= final_threshold).astype(np.int8))

    fold_csv = out_dir / "cv_fold_split_summary.csv"
    metrics_csv = out_dir / "results_summary.csv"
    config_json = out_dir / "run_config.json"
    means_csv = out_dir / "global_normalization_stats.csv"
    model_path = out_dir / "rf_global_model.joblib"

    fold_df.to_csv(fold_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "mean": [float(means[c]) for c in RAW_FEATURES],
            "std": [float(stds[c]) for c in RAW_FEATURES],
        }
    ).to_csv(means_csv, index=False)
    if final_model is not None:
        joblib.dump(final_model, model_path)

    with config_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "positive_class": positive_class,
                "window_seconds": args.window_seconds,
                "stride_seconds": args.stride_seconds,
                "threshold_strategy": args.threshold_strategy,
                "balance_train": args.balance_train,
                "discard_nurses": sorted(discard),
                "num_rows_all": int(len(df_all)),
                "num_nurses": int(df_all["nurse_id"].nunique()),
                "num_nurse_day_groups": int(len(group_keys)),
                "num_valid_folds": int(len(metrics_df)),
                "final_threshold": final_threshold,
                "final_model_train_metrics": final_train_metrics,
            },
            f,
            indent=2,
        )

    avg = metrics_df[
        ["test_accuracy", "test_balanced_accuracy", "test_f1", "test_precision", "test_recall"]
    ].mean(axis=0)
    print("\n=== Global LONO-Day CV Average ===")
    print(avg.to_string())
    print(f"\nWrote: {fold_csv}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {means_csv}")
    print(f"Wrote: {config_json}")
    if final_model is not None:
        print(f"Wrote: {model_path}")

    return {
        "positive_class": positive_class,
        "test_accuracy": float(avg["test_accuracy"]),
        "test_balanced_accuracy": float(avg["test_balanced_accuracy"]),
        "test_f1": float(avg["test_f1"]),
        "test_precision": float(avg["test_precision"]),
        "test_recall": float(avg["test_recall"]),
        "num_valid_folds": int(len(metrics_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global RF with global z-score normalization and leave-one-nurse-day-out validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Aditya"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sliding_window_rf"))
    parser.add_argument("--positive-class", choices=["stressed", "unstressed"], default="stressed")
    parser.add_argument("--run-both-targets", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--threshold-strategy", choices=["fixed_0_5", "balanced_acc"], default="balanced_acc")
    parser.add_argument("--balance-train", choices=["none", "undersample"], default="undersample")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--discard-nurses", nargs="*", default=["CE", "EG"])
    parser.add_argument("--min-train-windows", type=int, default=200)
    parser.add_argument("--min-test-windows", type=int, default=100)
    parser.add_argument("--min-train-c0", type=int, default=30)
    parser.add_argument("--min-train-c1", type=int, default=30)
    parser.add_argument("--min-test-c0", type=int, default=20)
    parser.add_argument("--min-test-c1", type=int, default=20)
    parser.add_argument("--min-train-c0-rate", type=float, default=0.05)
    parser.add_argument("--min-train-c1-rate", type=float, default=0.05)
    parser.add_argument("--min-test-c0-rate", type=float, default=0.05)
    parser.add_argument("--min-test-c1-rate", type=float, default=0.05)
    parser.add_argument("--max-class-ratio-diff", type=float, default=0.20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_both_targets:
        stressed_out = args.out_dir / "stressed"
        unstressed_out = args.out_dir / "unstressed"
        stressed_summary = run_global_lono_day(args, positive_class="stressed", out_dir=stressed_out)
        unstressed_summary = run_global_lono_day(args, positive_class="unstressed", out_dir=unstressed_out)

        comparison_df = pd.DataFrame([stressed_summary, unstressed_summary]).sort_values("positive_class")
        comparison_csv = args.out_dir / "target_comparison_summary.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\nWrote: {comparison_csv}")
        print("\n=== Target Comparison ===")
        print(comparison_df.to_string(index=False))
    else:
        run_global_lono_day(args, positive_class=args.positive_class, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
