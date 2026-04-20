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
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
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
EPOCHS        = 5
PATIENCE      = 5          # early-stopping patience
TRAIN_FRAC    = 0.7        # fraction of days used for training
VAL_FRAC      = 0.15       # remainder goes to test
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


def day_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC,
              val_frac: float = VAL_FRAC):
    """Split days into train / val / test.

    Test day is chosen as the day whose stress ratio is closest to 0.5
    (most balanced), so the test set always contains both classes.
    Val day is the most balanced among the remaining days.
    Train gets everything else.
    """
    days = sorted(df["date"].unique())

    # Balance score: closeness to 0.5 stress ratio (lower = more balanced)
    balance = df.groupby("date")["label_binary"].mean()
    scored  = sorted(days, key=lambda d: abs(balance[d] - 0.5))

    test_day = scored[0]                          # most balanced day → test
    remaining = [d for d in scored if d != test_day]

    val_day   = remaining[0] if remaining else None   # next most balanced → val
    train_days = [d for d in days if d not in {test_day, val_day}]

    train_df = df[df["date"].isin(train_days)]
    val_df   = df[df["date"].isin([val_day])]   if val_day   else df.iloc[:0]
    test_df  = df[df["date"].isin([test_day])]

    return train_df, val_df, test_df


def normalize_nurse(train_df, val_df, test_df, features=FEATURES):
    """Fit a StandardScaler on train split; transform all three splits."""
    scaler = StandardScaler()
    # Exclude time_progress from scaler (already in [0,1])
    scale_cols = [f for f in features if f != "time_progress"]

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols]   = scaler.transform(val_df[scale_cols])
    test_df[scale_cols]  = scaler.transform(test_df[scale_cols])

    return train_df, val_df, test_df, scaler


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

def compute_class_weights(df: pd.DataFrame, minority_boost: float = 2.0):
    """Inverse-frequency class weights with an extra boost for the minority class.

    minority_boost: multiplier applied on top of inverse-freq for class 0 (no-stress).
    Increase if the model still ignores no-stress after oversampling.
    """
    counts  = df[TARGET].value_counts().sort_index()
    total   = counts.sum()
    w = [total / (2 * counts.get(i, 1)) for i in range(2)]
    w[0] *= minority_boost      # extra penalty for missing no-stress
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


def train_nurse(nurse_id, train_df, val_df, test_df):
    """Full training loop for one nurse. Returns per-nurse metrics dict."""

    train_loader = make_loader(train_df, shuffle=True,  oversample=True)
    val_loader   = make_loader(val_df,   shuffle=False, oversample=False)
    test_loader  = make_loader(test_df,  shuffle=False, oversample=False)

    if train_loader is None:
        print(f"  [Nurse {nurse_id}] Not enough train data — skipping.")
        return None

    weights   = compute_class_weights(train_df)
    model     = StressLSTM(input_size=len(FEATURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_val_f1   = -1
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)

        if val_loader:
            val_loss, val_acc, _, val_y, _ = eval_epoch(model, val_loader, criterion)
            val_preds = eval_epoch(model, val_loader, criterion)[2]
            val_f1    = f1_score(val_y, val_preds, average="binary", zero_division=0)
            scheduler.step(val_loss)
        else:
            val_loss, val_acc, val_f1 = tr_loss, tr_acc, 0.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # ── Evaluate on test set ──────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    save_path = os.path.join(RESULTS_DIR, f"nurse_{nurse_id}_lstm.pt")
    torch.save(model.state_dict(), save_path)

    if test_loader is None:
        print(f"  [Nurse {nurse_id}] No test data.")
        return {"nurse_id": nurse_id, "note": "no test data"}

    _, test_acc, test_preds, test_y, test_probs = eval_epoch(
        model, test_loader, criterion
    )

    report = classification_report(test_y, test_preds,
                                   target_names=["no-stress", "stress"],
                                   output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(test_y, test_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(test_y, test_preds)
    print(f"\n  === Nurse {nurse_id} Test Results ===")
    print(f"  Acc={test_acc:.3f}  AUC={auc:.3f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(test_y, test_preds,
                                 target_names=["no-stress", "stress"],
                                 zero_division=0))

    return {
        "nurse_id":  nurse_id,
        "accuracy":  test_acc,
        "auc":       auc,
        "f1_macro":  report["macro avg"]["f1-score"],
        "f1_stress": report["stress"]["f1-score"],
        "precision_stress": report["stress"]["precision"],
        "recall_stress":    report["stress"]["recall"],
        "n_test_windows":   len(test_preds),
        "model_path":       save_path,
    }


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
            print(f"  [Nurse {nurse_id}] Skipped (no no-stress label).")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing Nurse {nurse_id}")
        print(f"{'='*60}")

        df = load_nurse(path)

        # Label distribution
        vc = df["label_binary"].value_counts().to_dict()
        print(f"  Rows: {len(df):,}  |  label dist: {vc}")

        # Day-based split
        train_df, val_df, test_df = day_split(df)
        n_days = df["date"].nunique()
        print(f"  Days: total={n_days}, "
              f"train={train_df['date'].nunique()}, "
              f"val={val_df['date'].nunique()}, "
              f"test={test_df['date'].nunique()}")

        if len(train_df) == 0:
            print("  Skipping — insufficient data.")
            continue

        # Per-nurse normalisation
        train_df, val_df, test_df, _ = normalize_nurse(train_df, val_df, test_df)

        # Train LSTM
        result = train_nurse(nurse_id, train_df, val_df, test_df)
        if result:
            all_results.append(result)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS ACROSS NURSES")
    print(f"{'='*60}")

    results_df = pd.DataFrame([r for r in all_results if "accuracy" in r])
    print(results_df[["nurse_id", "accuracy", "auc",
                       "f1_macro", "f1_stress"]].to_string(index=False))

    print("\n  Mean ± Std:")
    for col in ["accuracy", "auc", "f1_macro", "f1_stress"]:
        m, s = results_df[col].mean(), results_df[col].std()
        print(f"    {col:<20}: {m:.3f} ± {s:.3f}")

    out_csv = os.path.join(RESULTS_DIR, "nurse_results_summary.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Results saved to: {out_csv}")


if __name__ == "__main__":
    main()