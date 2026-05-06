#!/usr/bin/env python3
"""
Topic 79 — Model C: Neural Network (TCN-style)
===============================================

Strategy: A Temporal Convolutional Network (TCN) that processes the raw
OHLCV sequence directly, learning multi-scale temporal patterns for
volatility prediction. Uses dilated causal convolutions to capture both
short-term microstructure and longer-term regime dynamics.

Architecture:
    Input (30 bars × 5 OHLCV) → Conv1D blocks with dilations [1,2,4,8]
    → Global pooling → Dense layers → Volatility prediction

Trained on 2+ years of 1-minute BTC/USD data.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 120  # ~4 months (fast verification; increase for production)
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 30  # 30 minutes of 1-min bars
TARGET_BARS = 15
TARGET_TYPE = "volatility"

# Training config
EPOCHS = 60
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # early stopping — train longer

print("=" * 80)
print("Topic 79 — Model C: Neural Network (TCN-style, 2+ years)")
print("=" * 80)


# =============================================================================
# MODEL DEFINITION
# =============================================================================
class CausalConv1d(nn.Module):
    """Causal convolution: output at time t only depends on inputs at t and before."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove future padding (causal)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual TCN block with dilated causal convolution."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class VolatilityTCN(nn.Module):
    """
    TCN for volatility prediction.

    Input: (batch, seq_len, 5) — OHLCV bars
    Output: (batch, 1) — predicted volatility
    """

    def __init__(
        self,
        input_features=5,
        seq_len=30,
        hidden_channels=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_features, hidden_channels, 1)

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_channels, kernel_size, d, dropout)
            for d in dilations
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global average pooling
            nn.Flatten(),
            nn.Linear(hidden_channels, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),  # ensure non-negative output (volatility)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.tcn_blocks:
            x = block(x)
        return self.head(x)


# =============================================================================
# STEP 1: Initialize & Backfill
# =============================================================================
print("\n[1/5] Initializing workflow...")
from allora_forge_builder_kit.utils import get_api_key

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), ".allora_api_key")
)

workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    target_type=TARGET_TYPE,
    data_source="allora",
    api_key=api_key,
)
print(f"✅ {NUMBER_OF_INPUT_BARS} bars lookback, {TARGET_BARS}-min vol target")

print(f"\n[2/5] Backfilling {DAYS_OF_HISTORY} days...")
start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)
print("✅ Backfill complete")

# =============================================================================
# STEP 2: Prepare Data
# =============================================================================
print("\n[3/5] Preparing data for neural network...")
df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()

base_feature_cols = [col for col in df_all.columns if col.startswith("feature_")]
df_all = df_all.dropna(subset=base_feature_cols + ["target"])

# Reshape features into (samples, seq_len, 5) tensor
# Features are: feature_open_0..29, feature_high_0..29, etc.
n_samples = len(df_all)
seq_len = NUMBER_OF_INPUT_BARS
n_features = 5  # OHLCV

X_seq = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
for i in range(seq_len):
    X_seq[:, i, 0] = df_all[f"feature_open_{i}"].values
    X_seq[:, i, 1] = df_all[f"feature_high_{i}"].values
    X_seq[:, i, 2] = df_all[f"feature_low_{i}"].values
    X_seq[:, i, 3] = df_all[f"feature_close_{i}"].values
    X_seq[:, i, 4] = df_all[f"feature_volume_{i}"].values

y_all = df_all["target"].values.astype(np.float32)

print(f"✅ Dataset: {n_samples:,} samples, shape: ({seq_len}, {n_features})")

# =============================================================================
# STEP 3: Train with Walk-Forward CV
# =============================================================================
print("\n[4/5] Training neural network...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

tscv = TimeSeriesSplit(n_splits=2, gap=TARGET_BARS)
fold_predictions = np.full(n_samples, np.nan)

# Pre-allocate full tensors ONCE (zero-copy from numpy)
X_all_tensor = torch.from_numpy(X_seq)  # zero-copy, shares memory
y_all_tensor = torch.from_numpy(y_all).unsqueeze(1)

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
    print(f"\n   Fold {fold_idx + 1}/2: Train={len(train_idx):,}, Test={len(test_idx):,}")

    # Slice pre-allocated tensors (no copy, just views)
    X_train = X_all_tensor[train_idx]
    y_train = y_all_tensor[train_idx]
    X_test = X_all_tensor[test_idx]
    y_test = y_all_tensor[test_idx]

    # Simple index-based batching (faster than DataLoader for in-memory data)
    n_train = len(train_idx)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE

    model = VolatilityTCN(
        input_features=n_features,
        seq_len=seq_len,
        hidden_channels=96,  # increased capacity
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16),  # added dilation=16 for longer range
        dropout=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        # Train with manual batching (avoids DataLoader overhead)
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(idx)
        train_loss /= n_train
        scheduler.step()

        # Validate (in chunks to avoid OOM on large test sets)
        model.eval()
        with torch.no_grad():
            val_preds = []
            for vi in range(0, len(test_idx), BATCH_SIZE):
                vx = X_test[vi : vi + BATCH_SIZE].to(device)
                val_preds.append(model(vx))
            val_pred = torch.cat(val_preds)
            val_loss = nn.MSELoss()(val_pred, y_test.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter >= PATIENCE:
            print(f"      Epoch {epoch+1:2d}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}")

        if patience_counter >= PATIENCE:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    # Load best model and predict on test set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = []
        for vi in range(0, len(test_idx), BATCH_SIZE):
            vx = X_test[vi : vi + BATCH_SIZE].to(device)
            test_preds.append(model(vx))
        test_preds = torch.cat(test_preds).cpu().numpy().flatten()
    fold_predictions[test_idx] = test_preds

# Evaluate
valid_mask = ~np.isnan(fold_predictions)
evaluator = PerformanceEvaluator()
metrics = evaluator.evaluate(
    y_true=y_all[valid_mask],
    y_pred=fold_predictions[valid_mask],
)
print(f"\n✅ Neural network CV results: {metrics['num_passed']}/7 ({metrics['score']:.1%} — {metrics['grade']})")

# =============================================================================
# STEP 4: Train Final Model on All Data
# =============================================================================
print("\n[5/5] Training final model on all data...")

final_model = VolatilityTCN(
    input_features=n_features,
    seq_len=seq_len,
    hidden_channels=96,
    kernel_size=3,
    dilations=(1, 2, 4, 8, 16),
    dropout=0.05,
).to(device)

# Reuse pre-allocated tensors
n_all = len(X_all_tensor)
n_batches_all = (n_all + BATCH_SIZE - 1) // BATCH_SIZE

optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

for epoch in range(EPOCHS):
    final_model.train()
    perm = torch.randperm(n_all)
    epoch_loss = 0.0
    for bi in range(n_batches_all):
        idx = perm[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        xb = X_all_tensor[idx].to(device)
        yb = y_all_tensor[idx].to(device)
        pred = final_model(xb)
        loss = nn.MSELoss()(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= n_all
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:2d}: loss={epoch_loss:.8f}")

final_model.eval()
print(f"✅ Final model trained on {n_samples:,} samples")

# Move model to CPU for inference (deployment)
final_model = final_model.cpu()


def predict(nonce=None):
    """Predict 15-min BTC volatility using the TCN model."""
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")

    # Reshape to (1, seq_len, 5)
    x = np.zeros((1, NUMBER_OF_INPUT_BARS, 5), dtype=np.float32)
    row = live_row.iloc[0]
    for i in range(NUMBER_OF_INPUT_BARS):
        x[0, i, 0] = row[f"feature_open_{i}"]
        x[0, i, 1] = row[f"feature_high_{i}"]
        x[0, i, 2] = row[f"feature_low_{i}"]
        x[0, i, 3] = row[f"feature_close_{i}"]
        x[0, i, 4] = row[f"feature_volume_{i}"]

    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        vol = final_model(x_tensor).item()

    vol = max(0.0, vol)
    print(f"\nModel C (TCN) prediction: {vol:.6f} (15-min vol)")
    return vol


print("\n🧪 Testing...")
test_pred = predict()

with open("predict_79_model_c.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_79_model_c.pkl")
print(f"   Score: {metrics['score']:.1%} | Architecture: TCN (dilations 1,2,4,8)")
