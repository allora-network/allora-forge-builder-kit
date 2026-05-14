#!/usr/bin/env python3
"""
Feature Engineering Example for Allora Forge Builder Kit

This script demonstrates how to engineer features from base OHLCV features.
Base features are normalized candlesticks - here we show:
1. What base features look like (candlestick visualization)
2. How to engineer technical indicators (SMAs, returns, etc.)
3. How to validate features visually
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from allora_forge_builder_kit import AlloraMLWorkflow
import os

print("="*80)
print("Feature Engineering Example")
print("="*80)

# Configuration
TICKER = "btcusd"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 24
INTERVAL = "1h"
DAYS_OF_HISTORY = 30  # Just need recent data for visualization

# Read API key
api_key_path = os.path.join(os.path.dirname(__file__), '.allora_api_key')
with open(api_key_path, 'r') as f:
    api_key = f.read().strip()

# Initialize workflow
print("\n[1/3] Initializing workflow...")
workflow = AlloraMLWorkflow(
    tickers=[TICKER],
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    data_source="allora",
    api_key=api_key
)

# Get data
print(f"\n[2/3] Fetching {DAYS_OF_HISTORY} days of data...")
start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)

df = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
print(f"✅ Got {len(df):,} samples")

# =============================================================================
# Visualize Base Features
# =============================================================================
print("\n[3/3] Visualizing base features and engineered features...")

# Get one sample to visualize
sample = df.iloc[len(df)//2]  # Middle sample

# Extract OHLCV from base features (they're normalized)
# Base features are: feature_open_0, feature_high_0, ..., feature_close_23
opens = [sample[f'feature_open_{i}'] for i in range(NUMBER_OF_INPUT_BARS)]
highs = [sample[f'feature_high_{i}'] for i in range(NUMBER_OF_INPUT_BARS)]
lows = [sample[f'feature_low_{i}'] for i in range(NUMBER_OF_INPUT_BARS)]
closes = [sample[f'feature_close_{i}'] for i in range(NUMBER_OF_INPUT_BARS)]
volumes = [sample[f'feature_volume_{i}'] for i in range(NUMBER_OF_INPUT_BARS)]

# Denormalize for visualization (these are z-scored, so just for display)
# In practice, you'd work with normalized values
base_price = 100  # Arbitrary base for visualization
scale = 5  # Arbitrary scale

prices_open = base_price + np.array(opens) * scale
prices_high = base_price + np.array(highs) * scale
prices_low = base_price + np.array(lows) * scale
prices_close = base_price + np.array(closes) * scale

# =============================================================================
# Engineer Features
# =============================================================================

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros(len(prices))
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    ema[0] = np.mean(prices[:min(period, len(prices))])
    
    # Calculate EMA
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def engineer_features(df_row, num_bars):
    """
    Engineer features from base OHLCV features.
    All features are functions of the same row (no data leakage).
    
    Args:
        df_row: Single row with feature_open_0...feature_close_23
        num_bars: Number of input bars
    
    Returns:
        Dictionary of engineered features
    """
    features = {}
    
    # Extract close prices
    closes = np.array([df_row[f'feature_close_{i}'] for i in range(num_bars)])
    highs = np.array([df_row[f'feature_high_{i}'] for i in range(num_bars)])
    lows = np.array([df_row[f'feature_low_{i}'] for i in range(num_bars)])
    volumes = np.array([df_row[f'feature_volume_{i}'] for i in range(num_bars)])
    
    # 1. Simple Moving Averages
    features['sma_5'] = np.mean(closes[-5:]) if num_bars >= 5 else closes[-1]
    features['sma_10'] = np.mean(closes[-10:]) if num_bars >= 10 else closes[-1]
    features['sma_20'] = np.mean(closes[-20:]) if num_bars >= 20 else closes[-1]
    
    # 2. Returns over different horizons
    features['return_1h'] = closes[-1] - closes[-2] if num_bars >= 2 else 0
    features['return_6h'] = closes[-1] - closes[-7] if num_bars >= 7 else 0
    features['return_12h'] = closes[-1] - closes[-13] if num_bars >= 13 else 0
    
    # 3. Volatility (std of recent returns)
    if num_bars >= 6:
        recent_returns = np.diff(closes[-6:])
        features['volatility_5h'] = np.std(recent_returns)
    else:
        features['volatility_5h'] = 0
    
    # 4. High-Low spread (volatility indicator)
    features['hl_spread_mean'] = np.mean(highs - lows)
    features['hl_spread_recent'] = np.mean((highs - lows)[-5:]) if num_bars >= 5 else highs[-1] - lows[-1]
    
    # 5. Volume trend
    features['volume_mean'] = np.mean(volumes)
    features['volume_recent'] = np.mean(volumes[-5:]) if num_bars >= 5 else volumes[-1]
    
    # 6. Price momentum indicators
    features['price_distance_from_sma20'] = closes[-1] - features['sma_20']
    
    # 7. MACD (Moving Average Convergence Divergence)
    # MACD = 12-period EMA - 26-period EMA
    # Signal = 9-period EMA of MACD
    # Histogram = MACD - Signal
    if num_bars >= 12:
        ema_12 = calculate_ema(closes, 12)
        if num_bars >= 26:
            ema_26 = calculate_ema(closes, 26)
            macd_line = ema_12 - ema_26
            
            # Calculate signal line (9-period EMA of MACD)
            if num_bars >= 9:
                signal_line = calculate_ema(macd_line, 9)
                features['macd'] = macd_line[-1]
                features['macd_signal'] = signal_line[-1]
                features['macd_histogram'] = macd_line[-1] - signal_line[-1]
            else:
                features['macd'] = macd_line[-1]
                features['macd_signal'] = 0
                features['macd_histogram'] = macd_line[-1]
        else:
            # Not enough bars for full MACD, use simplified version
            features['macd'] = ema_12[-1] - closes[0]  # Approximate
            features['macd_signal'] = 0
            features['macd_histogram'] = features['macd']
    else:
        features['macd'] = 0
        features['macd_signal'] = 0
        features['macd_histogram'] = 0
    
    return features

# Apply feature engineering to sample
engineered = engineer_features(sample, NUMBER_OF_INPUT_BARS)

# Calculate SMAs for plotting
sma_5 = [np.mean(prices_close[max(0, i-4):i+1]) for i in range(len(prices_close))]
sma_10 = [np.mean(prices_close[max(0, i-9):i+1]) for i in range(len(prices_close))]
sma_20 = [np.mean(prices_close[max(0, i-19):i+1]) for i in range(len(prices_close))]

# Calculate MACD for plotting
ema_12 = calculate_ema(prices_close, 12)
ema_26 = calculate_ema(prices_close, 26)
macd_line = ema_12 - ema_26
signal_line = calculate_ema(macd_line, 9)
macd_histogram = macd_line - signal_line

# =============================================================================
# Plot
# =============================================================================

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Subplot 1: Candlestick chart with SMAs
x = np.arange(NUMBER_OF_INPUT_BARS)
colors = ['g' if prices_close[i] >= prices_open[i] else 'r' for i in range(len(x))]

for i in x:
    ax1.plot([i, i], [prices_low[i], prices_high[i]], color='black', linewidth=0.5)
    ax1.add_patch(plt.Rectangle(
        (i-0.4, min(prices_open[i], prices_close[i])),
        0.8,
        abs(prices_close[i] - prices_open[i]),
        facecolor=colors[i],
        edgecolor='black',
        linewidth=0.5
    ))

ax1.plot(x, sma_5, 'b-', label='SMA 5', linewidth=1.5, alpha=0.7)
ax1.plot(x, sma_10, 'orange', label='SMA 10', linewidth=1.5, alpha=0.7)
ax1.plot(x, sma_20, 'm-', label='SMA 20', linewidth=1.5, alpha=0.7)
ax1.set_ylabel('Price (normalized)', fontsize=10)
ax1.set_title(f'Base Features: {NUMBER_OF_INPUT_BARS} Hourly Candlesticks + SMAs\n' + 
              f'Sample from {sample["open_time"]}', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Volume
ax2.bar(x, volumes, color=['g' if c == 'g' else 'r' for c in colors], alpha=0.6)
ax2.set_ylabel('Volume (normalized)', fontsize=10)
ax2.set_title('Volume Profile', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: Returns
returns = np.diff(closes)
ax3.bar(x[1:], returns, color=['g' if r > 0 else 'r' for r in returns], alpha=0.6)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Return (normalized)', fontsize=10)
ax3.set_title('Bar-to-Bar Returns', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: MACD
ax4.plot(x, macd_line, 'b-', label='MACD Line', linewidth=1.5)
ax4.plot(x, signal_line, 'r-', label='Signal Line', linewidth=1.5)
hist_colors = ['g' if h > 0 else 'r' for h in macd_histogram]
ax4.bar(x, macd_histogram, color=hist_colors, alpha=0.3, label='Histogram')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Bar Index (0 = oldest, 23 = most recent)', fontsize=10)
ax4.set_ylabel('MACD (normalized)', fontsize=10)
ax4.set_title('MACD Indicator (12, 26, 9)', fontsize=11, fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('base_features_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to: base_features_visualization.png")

# =============================================================================
# Display Engineered Features
# =============================================================================

print("\n" + "="*80)
print("ENGINEERED FEATURES (from base features)")
print("="*80)
print("\nMoving Averages:")
print(f"  SMA 5:  {engineered['sma_5']:+.4f}")
print(f"  SMA 10: {engineered['sma_10']:+.4f}")
print(f"  SMA 20: {engineered['sma_20']:+.4f}")

print("\nReturns:")
print(f"  1-hour:  {engineered['return_1h']:+.4f}")
print(f"  6-hour:  {engineered['return_6h']:+.4f}")
print(f"  12-hour: {engineered['return_12h']:+.4f}")

print("\nVolatility:")
print(f"  5-hour volatility: {engineered['volatility_5h']:.4f}")
print(f"  HL spread mean:    {engineered['hl_spread_mean']:.4f}")
print(f"  HL spread recent:  {engineered['hl_spread_recent']:.4f}")

print("\nVolume:")
print(f"  Mean volume:   {engineered['volume_mean']:.4f}")
print(f"  Recent volume: {engineered['volume_recent']:.4f}")

print("\nMomentum:")
print(f"  Distance from SMA20: {engineered['price_distance_from_sma20']:+.4f}")

print("\nMACD:")
print(f"  MACD Line:      {engineered['macd']:+.4f}")
print(f"  Signal Line:    {engineered['macd_signal']:+.4f}")
print(f"  MACD Histogram: {engineered['macd_histogram']:+.4f}")
if engineered['macd_histogram'] > 0:
    print(f"  → Bullish signal (MACD > Signal)")
else:
    print(f"  → Bearish signal (MACD < Signal)")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("✅ Base features are normalized OHLCV candlesticks")
print("✅ All engineered features come from the SAME ROW (no data leakage)")
print("✅ SMAs, returns, volatility help capture price dynamics")
print("✅ These features can replace or augment base features in modeling")
print("\n💡 Next steps:")
print("   1. Add these engineered features to your workflow")
print("   2. Test if they improve model performance")
print("   3. Iterate on feature engineering based on results")
print("="*80)

