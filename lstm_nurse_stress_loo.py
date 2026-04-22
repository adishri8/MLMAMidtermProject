"""
Per-Nurse LSTM Stress Prediction Pipeline
==========================================
- Splits data by day
- Adds time_progress feature (0→1 within each day)
- Normalizes features per nurse (fit on train days, transform all)
- Binarizes labels: {0} → 0 (no-stress), {1, 2} → 1 (stress)
- Trains an LSTM per nurse with sequence windowing
- Evaluates per nurse and aggregates results
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve
)
from sklearn.model_selection import GroupShuffleSplit

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
SEQ_LEN       = 200        # ~6 s of 32 Hz data
STRIDE        = 50         # step between windows
BATCH_SIZE    = 64
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
LR            = 1e-3
EPOCHS        = 30
PATIENCE      = 5          # early-stopping patience
TRAIN_FRAC    = 0.7        # fraction of days used for training
VAL_FRAC      = 0.15       # remainder goes to test
# Acceptable stress ratio per day: days outside [MIN, MAX] are dropped
MIN_STRESS_RATIO = 0.20    # at least 20% stress
MAX_STRESS_RATIO = 0.80    # at most 80% stress (i.e. at least 20% no-stress)
DATA_DIR      = "data/Aditya"
RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["acc_mag", "EDA", "HR", "TEMP", "time_progress"]
TARGET   = "label_binary"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_nurse(path: str) -> pd.DataFrame:
    """Load one nurse CSV and apply all preprocessing steps."""
    df = pd.read_csv(path, parse_dates=["datetime"])

    # ── Binarize labels: 0 → no-stress, 1 & 2 → stress ──────────────────────
    df["label_binary"] = (df["label"] > 0).astype(int)

    # ── Extract date (day) for day-based splitting ────────────────────────────
    df["date"] = df["datetime"].dt.date

    # ── Add time_progress: fraction of the day elapsed (0 → 1) ───────────────
    df = df.sort_values("datetime").reset_index(drop=True)
    df["time_progress"] = (
        df.groupby("date")["time"]
          .transform(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-9))
    )

    return df


def filter_balanced_days(df: pd.DataFrame,
                         min_ratio: float = MIN_STRESS_RATIO,
                         max_ratio: float = MAX_STRESS_RATIO) -> pd.DataFrame:
    """Keep only days whose stress ratio falls within [min_ratio, max_ratio].

    Returns the filtered DataFrame and prints a per-day breakdown.
    """
    balance  = df.groupby("date")["label_binary"].mean()
    all_days = sorted(df["date"].unique())

    kept, dropped = [], []
    print(f"  {'DATE':<12} {'STRESS%':>8}  STATUS")
    print(f"  {'-'*35}")
    for d in all_days:
        ratio = balance[d]
        pct   = ratio * 100
        if min_ratio <= ratio <= max_ratio:
            kept.append(d)
            print(f"  {str(d):<12} {pct:>7.1f}%  ✓ kept")
        else:
            dropped.append(d)
            reason = "too little stress" if ratio < min_ratio else "too much stress"
            print(f"  {str(d):<12} {pct:>7.1f}%  ✗ dropped ({reason})")
    print(f"  → {len(kept)} / {len(all_days)} days kept "
          f"(ratio window {min_ratio*100:.0f}–{max_ratio*100:.0f}%)")

    return df[df["date"].isin(kept)].copy()


def normalize_nurse(train_df, test_df, features=FEATURES):
    """Fit a StandardScaler on train split; transform train + test."""
    scaler     = StandardScaler()
    scale_cols = [f for f in features if f != "time_progress"]

    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols]  = scaler.transform(test_df[scale_cols])

    return train_df, test_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEQUENCE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset over a single nurse's data."""

    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN,
                 stride: int = STRIDE):
        data   = df[FEATURES].values.astype(np.float32)
        labels = df[TARGET].values.astype(np.int64)

        self.X, self.y = [], []
        for start in range(0, len(data) - seq_len + 1, stride):
            end = start + seq_len
            self.X.append(data[start:end])
            # majority vote over the window
            window_labels = labels[start:end]
            self.y.append(int(window_labels.mean() >= 0.5))

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def make_loader(df, seq_len=SEQ_LEN, stride=STRIDE,
                batch_size=BATCH_SIZE, shuffle=True, oversample=False):
    ds = SequenceDataset(df, seq_len, stride)
    if len(ds) == 0:
        return None

    if oversample and shuffle:
        # WeightedRandomSampler: give each window a weight inverse to its class freq
        labels  = ds.y
        counts  = np.bincount(labels, minlength=2).astype(float)
        counts  = np.where(counts == 0, 1, counts)          # avoid div-by-zero
        w_per_class = 1.0 / counts
        weights = torch.tensor(w_per_class[labels], dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(ds, batch_size=batch_size,
                          sampler=sampler, drop_last=False)

    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL
# ─────────────────────────────────────────────────────────────────────────────

class StressLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)          # binary: no-stress vs stress
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last time-step


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame, minority_boost: float = 2.0,
                          max_weight: float = 20.0):
    """Inverse-frequency class weights, capped to avoid numerical explosion.

    minority_boost : extra multiplier for no-stress on top of inverse-freq.
    max_weight     : hard ceiling — prevents absurd values (e.g. 165k) when
                     the minority class is near-absent in the train split.
    """
    counts  = df[TARGET].value_counts().sort_index()
    total   = counts.sum()
    w = [total / (2 * counts.get(i, 1)) for i in range(2)]
    w[0] = min(w[0] * minority_boost, max_weight)
    w[1] = min(w[1], max_weight)
    weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights → no-stress: {w[0]:.2f}  stress: {w[1]:.2f}")
    
    return weights


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct    += (preds == y).sum().item()
        n          += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total_loss / n, correct / n,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


@torch.no_grad()
def permutation_importance(model, test_loader, criterion,
                           n_repeats: int = 10) -> dict:
    """Measure feature importance by permuting each feature and recording
    the drop in macro-F1 on the test set. Larger drop = more important feature.
    """
    model.eval()

    def get_f1(loader):
        all_preds, all_labels = [], []
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        return f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Collect all test tensors once to avoid re-iterating the loader
    all_X = torch.cat([X for X, _ in test_loader], dim=0)   # (N, seq_len, n_feat)
    all_y = torch.cat([y for _, y in test_loader], dim=0)

    base_ds     = torch.utils.data.TensorDataset(all_X, all_y)
    base_loader = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=False)
    base_f1     = get_f1(base_loader)

    importance = {}
    for fi, feat_name in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            X_perm = all_X.clone()
            # shuffle this feature across all windows & timesteps
            idx = torch.randperm(X_perm.shape[0])
            X_perm[:, :, fi] = X_perm[idx, :, fi]
            perm_ds     = torch.utils.data.TensorDataset(X_perm, all_y)
            perm_loader = DataLoader(perm_ds, batch_size=BATCH_SIZE, shuffle=False)
            drops.append(base_f1 - get_f1(perm_loader))
        importance[feat_name] = float(np.mean(drops))

    return importance


def print_split_balance(train_df, test_df, fold_i, n_folds):
    """Print train/test split balance for one LODO fold."""
    print(f"  Fold {fold_i+1}/{n_folds}  "
          f"{'Split':<6} {'Days':>5} {'No-stress':>10} {'Stress':>8} {'Stress%':>8}")
    print(f"  {'-'*55}")
    for name, df in [("train", train_df), ("test", test_df)]:
        vc   = df[TARGET].value_counts().to_dict()
        n0, n1 = vc.get(0, 0), vc.get(1, 0)
        pct  = 100 * n1 / (n0 + n1) if (n0 + n1) > 0 else 0
        print(f"  {'':8} {name:<6} {df['date'].nunique():>5} {n0:>10,} {n1:>8,} {pct:>7.1f}%")
    print()


def train_one_fold(nurse_id, fold_i, n_folds, train_df, test_df):
    """Train and evaluate one LODO fold. Returns a metrics dict."""

    print_split_balance(train_df, test_df, fold_i, n_folds)

    train_loader = make_loader(train_df, shuffle=True,  oversample=True)
    test_loader  = make_loader(test_df,  shuffle=False, oversample=False)

    if train_loader is None:
        print(f"  Fold {fold_i+1}: not enough train windows — skipping.")
        return None

    weights   = compute_class_weights(train_df)
    model     = StressLSTM(input_size=len(FEATURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_f1    = -1
    best_state = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        # Use train loss for scheduler since there's no separate val set in LODO
        scheduler.step(tr_loss)

        if epoch % 5 == 0 or epoch == 1:
            # Quick train-set F1 for progress monitoring only
            _, _, tr_preds, tr_y, _ = eval_epoch(model, train_loader, criterion)
            tr_f1 = f1_score(tr_y, tr_preds, average="macro", zero_division=0)
            print(f"  Epoch {epoch:02d} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_f1={tr_f1:.3f}")

            if tr_f1 > best_f1:
                best_f1    = tr_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

    if best_state:
        model.load_state_dict(best_state)

    if test_loader is None:
        return None

    _, test_acc, test_preds, test_y, test_probs = eval_epoch(
        model, test_loader, criterion
    )

    report = classification_report(test_y, test_preds,
                                   target_names=["no-stress", "stress"],
                                   output_dict=True, zero_division=0)
    try:
        pr_auc = average_precision_score(test_y, test_probs)
    except ValueError:
        pr_auc = float("nan")

    perm_imp = permutation_importance(model, test_loader, criterion, n_repeats=10)
    
    test_day = test_df["date"].iloc[0]
    cm = confusion_matrix(test_y, test_preds)
    print(f"  Fold {fold_i+1} test day={test_day} | "
          f"Acc={test_acc:.3f} PR-AUC={pr_auc:.3f}")
    print(f"  CM: {cm.tolist()}")
    print(classification_report(test_y, test_preds,
                                 target_names=["no-stress", "stress"],
                                 zero_division=0))
    
    return {
        "fold":               fold_i,
        "test_day":           str(test_day),
        "accuracy":           test_acc,
        "pr_auc":             pr_auc,
        "f1_macro":           report["macro avg"]["f1-score"],
        "precision_macro":    report["macro avg"]["precision"],
        "recall_macro":       report["macro avg"]["recall"],
        "f1_nostress":        report["no-stress"]["f1-score"],
        "precision_nostress": report["no-stress"]["precision"],
        "recall_nostress":    report["no-stress"]["recall"],
        "f1_stress":          report["stress"]["f1-score"],
        "precision_stress":   report["stress"]["precision"],
        "recall_stress":      report["stress"]["recall"],
        "n_test_windows":     len(test_preds),
        "perm_importance":    perm_imp,
    }


def train_nurse_lodo(nurse_id, df):
    """Leave-One-Day-Out cross-validation for one nurse.

    For each balanced day d:
      - test  = day d
      - train = all other balanced days
    Returns averaged metrics across folds.
    """
    days   = sorted(df["date"].unique())
    n_days = len(days)

    if n_days < 2:
        print(f"  [Nurse {nurse_id}] Only {n_days} balanced day(s) — skipping LODO.")
        return None

    print(f"  Running LODO over {n_days} balanced days...")
    fold_results = []

    for fold_i, test_day in enumerate(days):
        train_days = [d for d in days if d != test_day]
        train_df   = df[df["date"].isin(train_days)].copy()
        test_df    = df[df["date"] == test_day].copy()

        # Normalize: fit on train days only
        train_df, test_df, _ = normalize_nurse(train_df, test_df)

        result = train_one_fold(nurse_id, fold_i, n_days, train_df, test_df)
        if result:
            fold_results.append(result)

    if not fold_results:
        return None

    # ── Average metrics across folds ─────────────────────────────────────────
    metric_keys = [
        "accuracy", "pr_auc",
        "f1_macro", "precision_macro", "recall_macro",
        "f1_nostress", "precision_nostress", "recall_nostress",
        "f1_stress",   "precision_stress",   "recall_stress",
    ]
    avg = {k: float(np.nanmean([r[k] for r in fold_results])) for k in metric_keys}
    std = {f"{k}_std": float(np.nanstd([r[k] for r in fold_results])) for k in metric_keys}

    # Average permutation importance
    imp_df  = pd.DataFrame([r["perm_importance"] for r in fold_results])
    avg_imp = imp_df.mean().to_dict()

    save_path = os.path.join(RESULTS_DIR, f"nurse_{nurse_id}_lodo_folds.csv")
    pd.DataFrame(fold_results).drop(
        columns=["perm_importance"], errors="ignore"
    ).to_csv(save_path, index=False)

    print(f"=== Nurse {nurse_id} LODO Summary ({len(fold_results)} folds) ===")
    print(f"  PR-AUC:       {avg['pr_auc']:.3f} ± {std['pr_auc_std']:.3f}")
    print(f"  F1 no-stress: {avg['f1_nostress']:.3f} ± {std['f1_nostress_std']:.3f}")
    print(f"  F1 stress:    {avg['f1_stress']:.3f} ± {std['f1_stress_std']:.3f}")
    print(f"  F1 macro:     {avg['f1_macro']:.3f} ± {std['f1_macro_std']:.3f}")
    print(f"  Fold results → {save_path}")

    return {"nurse_id": nurse_id, "n_folds": len(fold_results),
            **avg, **std, "perm_importance": avg_imp}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "processed_nurse_*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No nurse CSVs found in '{DATA_DIR}'. "
            "Run the preprocessing script first."
        )
        
    print(f"Found {len(csv_files)} nurse files.\n")

    DISCARD_NURSES = {"CE", "EG", "6D"}   # CE/EG: no label-0 samples; 6D: only 1 day

    # ── Dataset overview ──────────────────────────────────────────────────────
    print(f"{'NURSE':<12} {'ROWS':>10} {'DAYS':>6} {'NO-STRESS':>12} {'STRESS':>10} {'STRESS%':>9}")
    print("-" * 65)
    for p in csv_files:
        nid = os.path.basename(p).replace("processed_nurse_", "").replace(".csv", "")
        if nid in DISCARD_NURSES:
            print(f"{nid:<12} {'(skipped)':>10}")
            continue
        _df = pd.read_csv(p, usecols=["datetime", "label"], parse_dates=["datetime"])
        _df["label_binary"] = (_df["label"] > 0).astype(int)
        _df["date"] = _df["datetime"].dt.date
        n_rows  = len(_df)
        n_days  = _df["date"].nunique()
        vc      = _df["label_binary"].value_counts().to_dict()
        n0, n1  = vc.get(0, 0), vc.get(1, 0)
        pct     = 100 * n1 / n_rows if n_rows else 0
        print(f"{nid:<12} {n_rows:>10,} {n_days:>6} {n0:>12,} {n1:>10,} {pct:>8.1f}%")
    print("-" * 65)
    print()

    all_results = []

    for path in csv_files:
        nurse_id = os.path.basename(path).replace("processed_nurse_", "").replace(".csv", "")

        if nurse_id in DISCARD_NURSES:
            print(f"  [Nurse {nurse_id}] Skipped.")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing Nurse {nurse_id}")
        print(f"{'='*60}")

        df = load_nurse(path)
        vc = df["label_binary"].value_counts().to_dict()
        print(f"  Rows: {len(df):,}  |  label dist: {vc}")

        # ── Filter to balanced days only ──────────────────────────────────────
        df_bal = filter_balanced_days(df)

        if df_bal["date"].nunique() < 2:
            print(f"  [Nurse {nurse_id}] Fewer than 2 balanced days — skipping.")
            continue

        # ── LODO cross-validation ─────────────────────────────────────────────
        result = train_nurse_lodo(nurse_id, df_bal)
        if result:
            all_results.append(result)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS ACROSS NURSES")
    print(f"{'='*60}")

    results_df = pd.DataFrame([r for r in all_results if "accuracy" in r])

    # ── Per-nurse table ───────────────────────────────────────────────────────
    display_cols = [
        "nurse_id", "accuracy", "pr_auc",
        "precision_nostress", "recall_nostress", "f1_nostress",
        "precision_stress",   "recall_stress",   "f1_stress",
        "precision_macro",    "recall_macro",     "f1_macro",
    ]
    print(results_df[display_cols].to_string(index=False))

    # ── Mean ± Std ────────────────────────────────────────────────────────────
    print("\n  Mean ± Std:")
    summary_cols = [
        "accuracy", "pr_auc",
        "precision_nostress", "recall_nostress", "f1_nostress",
        "precision_stress",   "recall_stress",   "f1_stress",
        "precision_macro",    "recall_macro",     "f1_macro",
    ]
    for col in summary_cols:
        m, s = results_df[col].mean(), results_df[col].std()
        print(f"    {col:<25}: {m:.3f} ± {s:.3f}")

    # ── Aggregate permutation importance across nurses ────────────────────────
    print("\n  Aggregate Permutation Feature Importance (mean F1 drop across nurses):")
    all_imp = pd.DataFrame([r["perm_importance"] for r in all_results
                            if "perm_importance" in r])
    imp_mean = all_imp.mean().sort_values(ascending=False)
    imp_std  = all_imp.std()
    for feat in imp_mean.index:
        print(f"    {feat:<15}: {imp_mean[feat]:+.4f} ± {imp_std[feat]:.4f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS_DIR, "nurse_results_summary.csv")
    results_df.drop(columns=["perm_importance", "model_path"],
                    errors="ignore").to_csv(out_csv, index=False)

    imp_csv = os.path.join(RESULTS_DIR, "feature_importance.csv")
    imp_df  = all_imp.copy()
    imp_df.index = [r["nurse_id"] for r in all_results if "perm_importance" in r]
    imp_df.to_csv(imp_csv)

    print(f"\n  Results saved to : {out_csv}")
    print(f"  Feature importance: {imp_csv}")


if __name__ == "__main__":
    main()