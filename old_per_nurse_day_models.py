"""
Legacy per-nurse per-day pipeline matching `per_nurse_day_models.py` behavior
except: no threshold tuning (fixed 0.5) and enforce class-ratio diff filter.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Reuse core utilities from the main pipeline to ensure behavior parity
from per_nurse_day_models import (
    RAW_FEATURES,
    load_nurse_csv,
    compute_nurse_normalization,
    apply_normalization,
    build_windows_for_day,
    generate_day_combos,
    has_min_class_mix,
    fit_and_predict,
    safe_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/Aditya'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/per_nurse_day_models_old'))
    parser.add_argument('--test-days-count', type=int, default=1)
    parser.add_argument('--min-train-windows', type=int, default=200)
    parser.add_argument('--min-test-windows', type=int, default=100)
    parser.add_argument('--min-train-c0', type=int, default=30)
    parser.add_argument('--min-train-c1', type=int, default=30)
    parser.add_argument('--min-test-c0', type=int, default=20)
    parser.add_argument('--min-test-c1', type=int, default=20)
    parser.add_argument('--min-train-c0-rate', type=float, default=0.05)
    parser.add_argument('--min-train-c1-rate', type=float, default=0.05)
    parser.add_argument('--min-test-c0-rate', type=float, default=0.05)
    parser.add_argument('--min-test-c1-rate', type=float, default=0.05)
    # no class-ratio-diff filtering for legacy script
    args = parser.parse_args()

    csv_files = sorted(Path(args.data_dir).glob('processed_nurse_*.csv'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []

    for csv_path in csv_files:
        nurse_id = csv_path.stem.replace('processed_nurse_', '')
        if nurse_id in ('CE', 'EG'):
            continue

        df = load_nurse_csv(csv_path)
        candidate_days = sorted(df['day'].unique().tolist())
        if len(candidate_days) < 2:
            continue

        means, stds = compute_nurse_normalization(df)
        df_norm = apply_normalization(df, means, stds)

        windows_by_day = {}
        for day, g in df_norm.groupby('day'):
            Xw, yw = build_windows_for_day(g, window_seconds=30.0, stride_seconds=5.0)
            if len(yw) > 0:
                windows_by_day[str(day)] = (Xw, yw)

        usable_days = sorted(windows_by_day.keys())
        if len(usable_days) < 2:
            continue

        test_day_combos = generate_day_combos(usable_days, args.test_days_count)
        if not test_day_combos:
            continue

        attempted = 0
        valid = 0

        for fold_idx, test_days in enumerate(test_day_combos, start=1):
            attempted += 1
            test_set = set(test_days)
            train_days = [d for d in usable_days if d not in test_set]
            if not train_days:
                continue

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

            valid += 1

            for model_name in ('decision_tree', 'random_forest'):
                y_proba, y_pred_default = fit_and_predict(model_name, X_train, y_train, X_test)
                # no tuning: use default 0.5 threshold (fit_and_predict returns pred at 0.5)
                m = safe_metrics(y_test, y_pred_default)
                m.update(
                    {
                        'nurse_id': nurse_id,
                        'fold_id': fold_idx,
                        'model': model_name,
                        'train_windows': int(len(y_train)),
                        'test_windows': int(len(y_test)),
                        'train_pos_rate': train_pos_rate,
                        'test_pos_rate': test_pos_rate,
                        'optimal_threshold': 0.5,
                        'test_days': '|'.join(sorted(test_days)),
                    }
                )
                metric_rows.append(m)

        # (optional) could write nurse-level summary

    if metric_rows:
        out_df = pd.DataFrame(metric_rows)
        out_df.to_csv(out_dir / 'cv_fold_model_metrics.csv', index=False)
        print('Wrote', out_dir / 'cv_fold_model_metrics.csv')
    else:
        print('No valid folds found.')


if __name__ == '__main__':
    main()
