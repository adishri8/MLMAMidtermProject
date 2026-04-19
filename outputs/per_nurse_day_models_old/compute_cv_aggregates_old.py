import pandas as pd
from pathlib import Path

base = Path(__file__).parent
csv_in = base / "cv_fold_model_metrics.csv"
csv_out = base / "cv_aggregate_summary_old.csv"
md_out = base / "CV_AGGREGATE_SUMMARY_old.md"

if not csv_in.exists():
    raise SystemExit(f"Missing input: {csv_in}")

df = pd.read_csv(csv_in)
metrics = ["accuracy","balanced_accuracy","f1_binary","f1_macro","precision","recall"]
for m in metrics:
    if m not in df.columns:
        df[m] = 0.0

rows = []
for model in sorted(df['model'].unique()):
    sub = df[df['model']==model]
    n = len(sub)
    if n==0:
        continue
    agg = {m: float(sub[m].mean()) for m in metrics}
    row = {
        'model': model,
        'n_folds': n,
        **{f'mean_{k}': agg[k] for k in metrics}
    }
    rows.append(row)

out_df = pd.DataFrame(rows)
out_df.to_csv(csv_out, index=False)

with md_out.open('w', encoding='utf-8') as f:
    f.write('# Legacy CV Aggregate Summary\n\n')
    f.write('Per-model averages from `outputs/per_nurse_day_models_old/cv_fold_model_metrics.csv`\n\n')
    f.write('| Model | n_folds | mean_accuracy | mean_balanced_accuracy | mean_f1_binary | mean_f1_macro | mean_precision | mean_recall |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for _, r in out_df.iterrows():
        f.write(f"| {r['model']} | {int(r['n_folds'])} | {r['mean_accuracy']:.4f} | {r['mean_balanced_accuracy']:.4f} | {r['mean_f1_binary']:.4f} | {r['mean_f1_macro']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} |\n")

print('Wrote:', csv_out)
print('Wrote:', md_out)
