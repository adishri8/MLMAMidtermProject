"""Compute SHAP feature importance for the per-nurse RF/DT folds.

The fold generation and window construction come from the shared per-nurse
pipeline; this script adds SHAP scoring and aggregation on top.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from RF_idealized_model_per_nurse import (
    RAW_FEATURES,
    apply_normalization,
    build_windows_for_day,
    compute_nurse_normalization,
    generate_day_combos,
    has_min_class_mix,
    load_nurse_csv,
    nurse_id_from_path,
)


@dataclass
class FoldData:
    """Store the data needed to fit and explain one fold."""
    pipeline: str
    nurse_id: str
    fold_id: int
    model_name: str
    train_windows: int
    test_windows: int
    test_days: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def build_window_feature_names() -> list[str]:
    """Return the exact feature order produced by the sliding-window builder."""
    names: list[str] = []
    stat_order = ["mean", "std", "min", "max", "last", "slope"]
    for stat in stat_order:
        for feat in RAW_FEATURES:
            names.append(f"{feat}_{stat}")
    names.append("day_time_sec")
    return names


def fit_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """Train the requested tree-based model for one fold."""
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
    return model


def compute_shap_abs_mean(
    model,
    X: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute mean absolute SHAP values for a sample of evaluation windows."""

    # Import SHAP lazily so the file still loads when the package is unavailable.
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Package 'shap' is required. Install it with: pip install shap"
        ) from exc

    if len(X) == 0:
        return np.zeros(shape=(X.shape[1],), dtype=np.float64)

    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    # SHAP return types vary by version/model:
    # - list of arrays [n_samples, n_features] per class
    # - array [n_samples, n_features]
    # - array [n_samples, n_features, n_outputs]
    if isinstance(shap_values, list):
        # Prefer positive class for binary classification.
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        # Use positive class if present, otherwise average over outputs.
        if values.shape[2] >= 2:
            values = values[:, :, 1]
        else:
            values = values.mean(axis=2)
    elif values.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP values shape: {values.shape}")

    return np.abs(values).mean(axis=0).astype(np.float64)


def collect_pipeline_folds(
    pipeline_name: str,
    csv_files: list[Path],
    test_days_count: int,
    window_seconds: float,
    stride_seconds: float,
    min_train_windows: int,
    min_test_windows: int,
    min_train_c0: int,
    min_train_c1: int,
    min_test_c0: int,
    min_test_c1: int,
    min_train_c0_rate: float,
    min_train_c1_rate: float,
    min_test_c0_rate: float,
    min_test_c1_rate: float,
    max_class_ratio_diff: float,
    discard_nurses: set[str],
    apply_class_ratio_filter: bool,
    max_folds_per_nurse: int | None,
    model_names: Iterable[str],
) -> list[FoldData]:
    """Build all valid folds for one pipeline configuration."""
    fold_data: list[FoldData] = []

    for csv_path in csv_files:
        nurse_id = nurse_id_from_path(csv_path)
        if nurse_id in discard_nurses:
            continue

        df = load_nurse_csv(csv_path)
        # Only keep nurses with enough distinct days for day-held-out validation.
        candidate_days = sorted(df["day"].unique().tolist())
        if len(candidate_days) < 2:
            continue

        means, stds = compute_nurse_normalization(df)
        df_norm = apply_normalization(df, means, stds)

        # Build one set of sliding windows per day so the folds stay day-grouped.
        windows_by_day: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for day, g in df_norm.groupby("day"):
            Xw, yw = build_windows_for_day(g, window_seconds, stride_seconds)
            if len(yw) > 0:
                windows_by_day[str(day)] = (Xw, yw)

        usable_days = sorted(windows_by_day.keys())
        if len(usable_days) < 2:
            continue

        test_day_combos = generate_day_combos(usable_days, test_days_count)
        if not test_day_combos:
            continue

        valid_fold_count = 0
        for fold_idx, test_days in enumerate(test_day_combos, start=1):
            test_day_set = set(test_days)
            train_days = [d for d in usable_days if d not in test_day_set]
            if not train_days:
                continue

            X_train = np.vstack([windows_by_day[d][0] for d in train_days])
            y_train = np.concatenate([windows_by_day[d][1] for d in train_days])
            X_test = np.vstack([windows_by_day[d][0] for d in test_days])
            y_test = np.concatenate([windows_by_day[d][1] for d in test_days])

            train_ok = has_min_class_mix(
                y=y_train,
                min_total=min_train_windows,
                min_c0=min_train_c0,
                min_c1=min_train_c1,
                min_c0_rate=min_train_c0_rate,
                min_c1_rate=min_train_c1_rate,
            )
            test_ok = has_min_class_mix(
                y=y_test,
                min_total=min_test_windows,
                min_c0=min_test_c0,
                min_c1=min_test_c1,
                min_c0_rate=min_test_c0_rate,
                min_c1_rate=min_test_c1_rate,
            )
            if not (train_ok and test_ok):
                continue

            if apply_class_ratio_filter:
                train_pos_rate = float(np.mean(y_train))
                test_pos_rate = float(np.mean(y_test))
                if abs(train_pos_rate - test_pos_rate) > max_class_ratio_diff:
                    continue

            valid_fold_count += 1
            if max_folds_per_nurse is not None and valid_fold_count > max_folds_per_nurse:
                break

            for model_name in model_names:
                fold_data.append(
                    FoldData(
                        pipeline=pipeline_name,
                        nurse_id=nurse_id,
                        fold_id=fold_idx,
                        model_name=model_name,
                        train_windows=int(len(y_train)),
                        test_windows=int(len(y_test)),
                        test_days="|".join(sorted(test_days)),
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )
                )

    return fold_data


def summarize_feature_importance(per_feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold SHAP and tree importances with test-window weighting."""
    grouped = (
        per_feature_rows.groupby(["pipeline", "model", "feature"], as_index=False)
        .agg(
            weighted_shap_importance=("weighted_shap", "sum"),
            weighted_tree_importance=("weighted_tree", "sum"),
            total_weight=("test_windows", "sum"),
            n_fold_rows=("fold_row", "sum"),
        )
        .sort_values(["pipeline", "model", "weighted_shap_importance"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    grouped["mean_shap_importance"] = grouped["weighted_shap_importance"] / grouped["total_weight"]
    grouped["mean_tree_importance"] = grouped["weighted_tree_importance"] / grouped["total_weight"]
    return grouped


def main() -> None:
    """Run the SHAP feature-importance workflow end to end."""
    parser = argparse.ArgumentParser(
        description="Compute SHAP feature importance for RF_idealized_model_per_nurse and RF_standard_model_per_nurse folds."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Aditya"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/RF_idealized_model_per_nurse_shap"))
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--test-days-count", type=int, default=1)
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
    parser.add_argument("--max-class-ratio-diff", type=float, default=0.15)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--max-folds-per-nurse", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["decision_tree", "random_forest"],
        choices=["decision_tree", "random_forest"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Build both pipelines from the same nurse CSV inputs.
    csv_files = sorted(args.data_dir.glob("processed_nurse_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No nurse CSV files found in {args.data_dir}")

    discard = set(args.discard_nurses)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_folds: list[FoldData] = []

    # Idealized pipeline keeps the class-ratio filter on.
    all_folds.extend(
        collect_pipeline_folds(
            pipeline_name="RF_idealized_model_per_nurse",
            csv_files=csv_files,
            test_days_count=args.test_days_count,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_train_windows=args.min_train_windows,
            min_test_windows=args.min_test_windows,
            min_train_c0=args.min_train_c0,
            min_train_c1=args.min_train_c1,
            min_test_c0=args.min_test_c0,
            min_test_c1=args.min_test_c1,
            min_train_c0_rate=args.min_train_c0_rate,
            min_train_c1_rate=args.min_train_c1_rate,
            min_test_c0_rate=args.min_test_c0_rate,
            min_test_c1_rate=args.min_test_c1_rate,
            max_class_ratio_diff=args.max_class_ratio_diff,
            discard_nurses=discard,
            apply_class_ratio_filter=True,
            max_folds_per_nurse=args.max_folds_per_nurse,
            model_names=args.models,
        )
    )

    # Legacy pipeline reuses the same folds but skips the class-ratio filter.
    all_folds.extend(
        collect_pipeline_folds(
            pipeline_name="RF_standard_model_per_nurse",
            csv_files=csv_files,
            test_days_count=args.test_days_count,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_train_windows=args.min_train_windows,
            min_test_windows=args.min_test_windows,
            min_train_c0=args.min_train_c0,
            min_train_c1=args.min_train_c1,
            min_test_c0=args.min_test_c0,
            min_test_c1=args.min_test_c1,
            min_train_c0_rate=args.min_train_c0_rate,
            min_train_c1_rate=args.min_train_c1_rate,
            min_test_c0_rate=args.min_test_c0_rate,
            min_test_c1_rate=args.min_test_c1_rate,
            max_class_ratio_diff=args.max_class_ratio_diff,
            discard_nurses=discard,
            apply_class_ratio_filter=False,
            max_folds_per_nurse=args.max_folds_per_nurse,
            model_names=args.models,
        )
    )

    if not all_folds:
        raise RuntimeError("No valid folds found for either pipeline.")

    # Save the exact folds before fitting SHAP so the results are reproducible.
    fold_summary_rows = [
        {
            "pipeline": f.pipeline,
            "nurse_id": f.nurse_id,
            "fold_id": f.fold_id,
            "model": f.model_name,
            "train_windows": f.train_windows,
            "test_windows": f.test_windows,
            "test_days": f.test_days,
        }
        for f in all_folds
    ]
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    fold_summary_df.to_csv(out_dir / "RF_idealized_model_per_nurse_shap_fold_manifest.csv", index=False)

    if args.dry_run:
        print("Dry-run only. Wrote fold manifest and exited.")
        print(f"Wrote: {out_dir / 'RF_idealized_model_per_nurse_shap_fold_manifest.csv'}")
        return

    # Fit each fold model and score the held-out windows.
    rng = np.random.default_rng(args.random_state)
    feature_names = build_window_feature_names()

    per_feature_rows: list[dict[str, object]] = []

    for fold in all_folds:
        model = fit_model(fold.model_name, fold.X_train, fold.y_train)

        shap_imp = compute_shap_abs_mean(
            model=model,
            X=fold.X_test,
            sample_size=args.sample_size,
            rng=rng,
        )
        tree_imp = getattr(model, "feature_importances_", np.zeros_like(shap_imp, dtype=np.float64))

        if len(shap_imp) != len(feature_names):
            raise RuntimeError(
                f"Feature length mismatch for {fold.pipeline}/{fold.nurse_id}/fold{fold.fold_id}: "
                f"got {len(shap_imp)} SHAP values, expected {len(feature_names)}"
            )

        for idx, feature in enumerate(feature_names):
            per_feature_rows.append(
                {
                    "pipeline": fold.pipeline,
                    "nurse_id": fold.nurse_id,
                    "fold_id": fold.fold_id,
                    "model": fold.model_name,
                    "feature": feature,
                    "shap_importance": float(shap_imp[idx]),
                    "tree_importance": float(tree_imp[idx]),
                    "test_windows": fold.test_windows,
                    "weighted_shap": float(shap_imp[idx]) * float(fold.test_windows),
                    "weighted_tree": float(tree_imp[idx]) * float(fold.test_windows),
                    "fold_row": 1,
                }
            )

    per_feature_df = pd.DataFrame(per_feature_rows)
    per_feature_path = out_dir / "RF_idealized_model_per_nurse_shap_per_fold_feature_importance.csv"
    per_feature_df.to_csv(per_feature_path, index=False)

    # Write both the weighted summary and a top-10 ranking for quick inspection.
    summary_df = summarize_feature_importance(per_feature_df)
    summary_path = out_dir / "RF_idealized_model_per_nurse_shap_feature_importance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    top10_df = (
        summary_df.sort_values(["pipeline", "model", "mean_shap_importance"], ascending=[True, True, False])
        .groupby(["pipeline", "model"], as_index=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10_path = out_dir / "RF_idealized_model_per_nurse_shap_feature_importance_top10.csv"
    top10_df.to_csv(top10_path, index=False)

    print(f"Wrote: {out_dir / 'RF_idealized_model_per_nurse_shap_fold_manifest.csv'}")
    print(f"Wrote: {per_feature_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {top10_path}")


if __name__ == "__main__":
    main()
