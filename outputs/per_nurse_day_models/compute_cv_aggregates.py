import pandas as pd
from pathlib import Path

base = Path(__file__).parent
csv_in = base / "cv_fold_model_metrics.csv"
csv_out = base / "cv_aggregate_summary.csv"
md_out = base / "CV_AGGREGATE_SUMMARY.md"

if not csv_in.exists():
    raise SystemExit(f"Missing input: {csv_in}")

df = pd.read_csv(csv_in)
# Ensure optimal_threshold present
if "optimal_threshold" not in df.columns:
    df["optimal_threshold"] = 0.5

models = []
rows = []
for model in sorted(df['model'].unique()):
    sub = df[df['model'] == model]
    n_folds = len(sub)
    if n_folds == 0:
        continue
    agg = sub[['accuracy','balanced_accuracy','f1_binary','f1_macro','precision','recall','optimal_threshold']].mean()
    row = {
        'model': model,
        'n_folds': n_folds,
        'mean_accuracy': float(agg['accuracy']),
        'mean_balanced_accuracy': float(agg['balanced_accuracy']),
        'mean_f1_binary': float(agg['f1_binary']),
        'mean_f1_macro': float(agg['f1_macro']),
        'mean_precision': float(agg['precision']),
        'mean_recall': float(agg['recall']),
        'mean_optimal_threshold': float(agg['optimal_threshold']),
    }
    rows.append(row)

agg_df = pd.DataFrame(rows)
agg_df.to_csv(csv_out, index=False)

with md_out.open('w', encoding='utf-8') as f:
    f.write('# CV Aggregate Summary\n\n')
    f.write('Per-model averages across valid folds.\n\n')
    # Write a simple markdown table without external dependencies
    f.write('| Model | n_folds | mean_accuracy | mean_balanced_accuracy | mean_f1_binary | mean_f1_macro | mean_precision | mean_recall | mean_optimal_threshold |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for _, r in agg_df.iterrows():
        f.write(f"| {r['model']} | {int(r['n_folds'])} | {r['mean_accuracy']:.4f} | {r['mean_balanced_accuracy']:.4f} | {r['mean_f1_binary']:.4f} | {r['mean_f1_macro']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {r['mean_optimal_threshold']:.2f} |\n")

print('Wrote:', csv_out)
print('Wrote:', md_out)
