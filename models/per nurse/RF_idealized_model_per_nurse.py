"""Per-nurse day-grouped RF/DT training helpers and the idealized pipeline.

The helper functions in this file are reused by the legacy per-nurse script and
the SHAP analysis so the sliding-window logic stays in one place.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


RAW_FEATURES = ["acc_mag", "EDA", "HR", "TEMP"]


def nurse_id_from_path(path: Path) -> str:
    """Convert a processed nurse filename into its short nurse ID."""
    return path.stem.replace("processed_nurse_", "")


def load_nurse_csv(csv_path: Path) -> pd.DataFrame:
    """Load a nurse CSV and add the day and label columns used downstream."""
    df = pd.read_csv(csv_path)
    required = {"datetime", "time", "acc_mag", "EDA", "HR", "TEMP", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["label_bin"] = (df["label"] > 0).astype(np.int8)
    df["day"] = df["datetime"].dt.strftime("%Y-%m-%d")
    day_start = df["datetime"].dt.floor("D")
    df["day_time_sec"] = (df["datetime"] - day_start).dt.total_seconds().astype(np.float32)
    return df


def infer_step_seconds(time_values: np.ndarray) -> float:
    """Infer the sample spacing from the time column."""
    if len(time_values) < 2:
        return 1.0
    diffs = np.diff(time_values)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))


def compute_nurse_normalization(df_source: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute per-nurse z-score statistics before windowing."""
    means = df_source[RAW_FEATURES].mean(axis=0)
    stds = df_source[RAW_FEATURES].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def apply_normalization(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Attach normalized feature columns for every raw sensor channel."""
    df = df.copy()
    for col in RAW_FEATURES:
        df[f"{col}_z"] = (df[col] - float(means[col])) / float(stds[col])
    return df


def build_windows_for_day(
    day_df: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize a single day into rolling windows with simple statistics."""
    feat_cols = [f"{c}_z" for c in RAW_FEATURES]
    X = day_df[feat_cols].to_numpy(dtype=np.float32)
    y = day_df["label_bin"].to_numpy(dtype=np.int8)
    t = day_df["day_time_sec"].to_numpy(dtype=np.float32)

    step = infer_step_seconds(t)
    window_steps = max(1, int(round(window_seconds / step)))
    stride_steps = max(1, int(round(stride_seconds / step)))

    if len(X) < window_steps:
        return np.empty((0, len(RAW_FEATURES) * 6 + 1), dtype=np.float32), np.empty((0,), dtype=np.int8)

    Xw: list[np.ndarray] = []
    yw: list[int] = []

    for start in range(0, len(X) - window_steps + 1, stride_steps):
        end = start + window_steps
        w = X[start:end]

        mean = w.mean(axis=0)
        std = w.std(axis=0)
        minv = w.min(axis=0)
        maxv = w.max(axis=0)
        last = w[-1]
        slope = (w[-1] - w[0]) / max(1, window_steps - 1)
        end_time = np.asarray([t[end - 1]], dtype=np.float32)

        Xw.append(np.concatenate([mean, std, minv, maxv, last, slope, end_time], axis=0))
        yw.append(int(y[end - 1]))

    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.int8)


def generate_day_combos(days: list[str], test_days_count: int) -> list[tuple[str, ...]]:
    """Enumerate the day combinations that can act as held-out test folds."""
    if test_days_count < 1:
        return []
    if len(days) <= test_days_count:
        return []
    return list(itertools.combinations(sorted(days), test_days_count))


def has_min_class_mix(
    y: np.ndarray,
    min_total: int,
    min_c0: int,
    min_c1: int,
    min_c0_rate: float,
    min_c1_rate: float,
) -> bool:
    """Reject folds that do not have enough windows or class balance."""
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


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute metrics without crashing on empty or single-class arrays."""
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_binary": 0.0,
            "f1_macro": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    if len(np.unique(y_true)) > 1:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    else:
        bal_acc = 0.0

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": bal_acc,
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def compute_per_day_normalized_stats(df_norm: pd.DataFrame, nurse_id: str) -> pd.DataFrame:
    """Summarize the normalized data day by day for inspection and QA."""
    rows: list[dict[str, Any]] = []
    for day, g in df_norm.groupby("day"):
        row: dict[str, Any] = {
            "nurse_id": nurse_id,
            "day": day,
            "n_samples": int(len(g)),
            "duration_sec": float((g["datetime"].max() - g["datetime"].min()).total_seconds()),
            "label_zero_rate": float(np.mean(g["label_bin"] == 0)),
            "label_one_rate": float(np.mean(g["label_bin"] == 1)),
        }
        for col in RAW_FEATURES:
            z = g[f"{col}_z"]
            row[f"{col}_z_mean"] = float(z.mean())
            row[f"{col}_z_std"] = float(z.std(ddof=0))
            row[f"{col}_z_min"] = float(z.min())
            row[f"{col}_z_max"] = float(z.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["nurse_id", "day"]).reset_index(drop=True)


def find_optimal_threshold_balanced_acc(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Pick the cutoff that maximizes balanced accuracy on the evaluation set."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_bal_acc = 0.0

    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_proba >= t).astype(np.int8)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(t)

    return best_threshold, best_bal_acc


def fit_and_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train the requested model and return probabilities plus hard labels."""
    if len(np.unique(y_train)) < 2:
        constant = int(y_train[0]) if len(y_train) > 0 else 1
        proba = np.full(shape=(len(X_test),), fill_value=float(constant), dtype=np.float32)
        return proba, np.full(shape=(len(X_test),), fill_value=constant, dtype=np.int8)

    if model_name == "decision_tree":
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X_train, y_train)
    proba_all = model.predict_proba(X_test)
    proba = proba_all[:, 1]
    pred = (proba >= 0.5).astype(np.int8)
    return proba, pred


def main() -> None:
    """Run the idealized per-nurse evaluation end to end."""
    parser = argparse.ArgumentParser(
        description="Train per-nurse per-day DT/RF models with day-grouped validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Aditya"))
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--discard-nurses", nargs="*", default=["CE", "EG"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/RF_idealized_model_per_nurse"))
    parser.add_argument("--test-days-count", type=int, default=1)
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
    parser.add_argument("--max-class-ratio-diff", type=float, default=0.15)
    args = parser.parse_args()

    # Load each nurse once, normalize within nurse, then build per-day windows.
    csv_files = sorted(args.data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No nurse CSV files found in {args.data_dir}")

    discard = set(args.discard_nurses)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    day_stats_all: list[pd.DataFrame] = []
    nurse_summary_rows: list[dict[str, Any]] = []

    # The idealized pipeline keeps the stricter class-ratio filter enabled.
    for csv_path in csv_files:
        nurse_id = nurse_id_from_path(csv_path)
        if nurse_id in discard:
            continue

        df = load_nurse_csv(csv_path)
        candidate_days = sorted(df["day"].unique().tolist())
        if len(candidate_days) < 2:
            print(f"[SKIP] Nurse {nurse_id}: not enough days for day-group CV.")
            continue

        means, stds = compute_nurse_normalization(df)
        df_norm = apply_normalization(df, means, stds)
        day_stats_all.append(compute_per_day_normalized_stats(df_norm, nurse_id))

        windows_by_day: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for day, g in df_norm.groupby("day"):
            Xw, yw = build_windows_for_day(g, args.window_seconds, args.stride_seconds)
            if len(yw) > 0:
                windows_by_day[str(day)] = (Xw, yw)

        usable_days = sorted(windows_by_day.keys())
        if len(usable_days) < 2:
            print(f"[SKIP] Nurse {nurse_id}: not enough usable days with windows.")
            continue

        test_day_combos = generate_day_combos(usable_days, args.test_days_count)
        if not test_day_combos:
            print(f"[SKIP] Nurse {nurse_id}: no candidate day combinations.")
            continue

        attempted_fold_count = 0
        valid_fold_count = 0

        for fold_idx, test_days in enumerate(test_day_combos, start=1):
            test_day_set = set(test_days)
            train_days = [d for d in usable_days if d not in test_day_set]
            if not train_days:
                continue

            attempted_fold_count += 1
            X_train = np.vstack([windows_by_day[d][0] for d in train_days])
            y_train = np.concatenate([windows_by_day[d][1] for d in train_days])
            X_test = np.vstack([windows_by_day[d][0] for d in test_days])
            y_test = np.concatenate([windows_by_day[d][1] for d in test_days])

            train_ok = has_min_class_mix(
                y=y_train,
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

            train_pos_rate = float(np.mean(y_train))
            test_pos_rate = float(np.mean(y_test))
            class_ratio_diff = abs(train_pos_rate - test_pos_rate)
            if class_ratio_diff > args.max_class_ratio_diff:
                continue

            valid_fold_count += 1
            fold_rows.append(
                {
                    "nurse_id": nurse_id,
                    "fold_id": fold_idx,
                    "train_days": "|".join(sorted(train_days)),
                    "test_days": "|".join(sorted(test_days)),
                    "num_train_days": len(train_days),
                    "num_test_days": len(test_days),
                    "train_windows": int(len(y_train)),
                    "test_windows": int(len(y_test)),
                    "train_c0": int(np.sum(y_train == 0)),
                    "train_c1": int(np.sum(y_train == 1)),
                    "test_c0": int(np.sum(y_test == 0)),
                    "test_c1": int(np.sum(y_test == 1)),
                    "train_pos_rate": train_pos_rate,
                    "test_pos_rate": test_pos_rate,
                }
            )

            for model_name in ["decision_tree", "random_forest"]:
                y_proba, y_pred_default = fit_and_predict(model_name, X_train, y_train, X_test)
                optimal_threshold, _ = find_optimal_threshold_balanced_acc(y_test, y_proba)
                y_pred_optimal = (y_proba >= optimal_threshold).astype(np.int8)

                m = safe_metrics(y_test, y_pred_optimal)
                m.update(
                    {
                        "nurse_id": nurse_id,
                        "fold_id": fold_idx,
                        "model": model_name,
                        "train_windows": int(len(y_train)),
                        "test_windows": int(len(y_test)),
                        "train_pos_rate": train_pos_rate,
                        "test_pos_rate": test_pos_rate,
                        "class_ratio_diff": class_ratio_diff,
                        "optimal_threshold": float(optimal_threshold),
                        "test_days": "|".join(sorted(test_days)),
                    }
                )
                metric_rows.append(m)

        if valid_fold_count == 0:
            print(
                f"[SKIP] Nurse {nurse_id}: no valid folds after class-mix filtering "
                f"({attempted_fold_count} attempted)."
            )
            continue

        nurse_summary_rows.append(
            {
                "nurse_id": nurse_id,
                "num_days_total": len(candidate_days),
                "num_days_with_windows": len(usable_days),
                "folds_attempted": attempted_fold_count,
                "folds_valid": valid_fold_count,
            }
        )
        print(f"[Nurse {nurse_id}] valid_folds={valid_fold_count}/{attempted_fold_count}")

    if not metric_rows:
        raise RuntimeError("No valid nurse folds remained after filtering. Relax class-mix thresholds.")

    # Persist fold-level outputs first so the evaluation split is traceable.
    fold_df = pd.DataFrame(fold_rows).sort_values(["nurse_id", "fold_id"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["nurse_id", "fold_id", "model"]).reset_index(drop=True)
    nurse_summary_df = pd.DataFrame(nurse_summary_rows).sort_values("nurse_id").reset_index(drop=True)
    day_stats_df = pd.concat(day_stats_all, axis=0).sort_values(["nurse_id", "day"]).reset_index(drop=True)

    fold_csv = args.out_dir / "RF_idealized_model_per_nurse_fold_split_summary.csv"
    metrics_csv = args.out_dir / "RF_idealized_model_per_nurse_fold_metrics.csv"
    nurse_summary_csv = args.out_dir / "RF_idealized_model_per_nurse_nurse_summary.csv"
    day_stats_csv = args.out_dir / "RF_idealized_model_per_nurse_day_normalized_stats.csv"
    config_json = args.out_dir / "RF_idealized_model_per_nurse_run_config.json"

    fold_df.to_csv(fold_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    nurse_summary_df.to_csv(nurse_summary_csv, index=False)
    day_stats_df.to_csv(day_stats_csv, index=False)

    with config_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "window_seconds": args.window_seconds,
                "stride_seconds": args.stride_seconds,
                "test_days_count": args.test_days_count,
                "min_train_windows": args.min_train_windows,
                "min_test_windows": args.min_test_windows,
                "min_train_c0": args.min_train_c0,
                "min_train_c1": args.min_train_c1,
                "min_test_c0": args.min_test_c0,
                "min_test_c1": args.min_test_c1,
                "min_train_c0_rate": args.min_train_c0_rate,
                "min_train_c1_rate": args.min_train_c1_rate,
                "min_test_c0_rate": args.min_test_c0_rate,
                "min_test_c1_rate": args.min_test_c1_rate,
                "max_class_ratio_diff": args.max_class_ratio_diff,
                "discard_nurses": sorted(discard),
                "num_nurses_modeled": int(metrics_df["nurse_id"].nunique()),
                "num_valid_folds": int(metrics_df[["nurse_id", "fold_id"]].drop_duplicates().shape[0]),
            },
            f,
            indent=2,
        )

    model_avg = (
        metrics_df.groupby("model", as_index=False)[
            ["accuracy", "balanced_accuracy", "f1_binary", "f1_macro", "precision", "recall"]
        ]
        .mean()
        .sort_values("model")
    )
    print("\n=== Average Across Valid Folds ===")
    print(model_avg.to_string(index=False))
    print(f"\nWrote: {fold_csv}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {nurse_summary_csv}")
    print(f"Wrote: {day_stats_csv}")
    print(f"Wrote: {config_json}")


if __name__ == "__main__":
    main()
