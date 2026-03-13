"""
Test to verify live features match historical features for the same timestamp.

This ensures that the transformations applied during training (historical features)
are identical to those applied during inference (live features).
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from allora_forge_builder_kit import AlloraMLWorkflow


def _load_api_key():
    key = os.environ.get("ALLORA_API_KEY")
    if key:
        return key
    for p in [Path("notebooks/.allora_api_key"), Path(".allora_api_key")]:
        if p.exists():
            content = p.read_text().strip()
            if content:
                return content
    return None


ALLORA_API_KEY = _load_api_key()
pytestmark = pytest.mark.skip(
    reason="Tests reference removed workflow methods (create_interval_bars, "
    "extract_features). Needs rewrite against current API (resample_ohlcv_polars, "
    "extract_features_polars, stand_alone_features_from_1min_bars)."
)


def test_live_vs_historical_features_5min():
    """Test that live features match historical features for 5-minute bars."""
    ticker = "btcusd"
    
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=12,  # 1 hour of 5-min bars
        target_bars=1,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 1] 5-minute bars: Live vs Historical Feature Consistency")
    print("="*80)
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=1)
    workflow.backfill(start=start)
    
    # Get historical features
    df_historical = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with complete features
    feature_cols = [col for col in df_historical.columns if col.startswith('feature_')]
    df_historical = df_historical.dropna(subset=feature_cols)
    
    if len(df_historical) == 0:
        pytest.skip("No complete historical features available")
    
    # Get a recent timestamp from historical data (not the absolute latest to ensure completeness)
    historical_timestamps = df_historical.xs(ticker, level='ticker').index
    test_timestamp = historical_timestamps[-10]  # 10 bars back from latest
    
    print(f"\nTest timestamp: {test_timestamp}")
    
    # Get historical features for this timestamp
    hist_features = df_historical.xs((ticker, test_timestamp))
    hist_feature_values = hist_features[feature_cols].values
    
    print(f"Historical features shape: {hist_feature_values.shape}")
    print(f"First 5 features: {hist_feature_values[:5]}")
    print(f"Last 5 features: {hist_feature_values[-5:]}")
    
    # Now simulate getting "live" features for the same timestamp
    # We'll load raw data up to that timestamp and extract features
    raw_data = workflow.load_raw(start=start, end=test_timestamp + timedelta(minutes=1))
    ticker_data = raw_data.xs(ticker, level='symbol')
    
    # Resample (same as historical)
    resampled = workflow.create_interval_bars(
        ticker_data.reset_index().set_index('open_time'),
        live_mode=False
    )
    
    # Extract features for the test timestamp
    live_features = workflow.extract_features(
        resampled,
        workflow.number_of_input_bars,
        [test_timestamp]
    )
    
    if live_features.empty:
        pytest.fail("No live features extracted")
    
    live_feature_values = live_features.iloc[0].values
    
    print(f"\nLive features shape: {live_feature_values.shape}")
    print(f"First 5 features: {live_feature_values[:5]}")
    print(f"Last 5 features: {live_feature_values[-5:]}")
    
    # Compare
    print(f"\n{'Feature':<20} {'Historical':<15} {'Live':<15} {'Diff':<15} {'Match'}")
    print("-"*80)
    
    mismatches = []
    for i, (hist_val, live_val) in enumerate(zip(hist_feature_values, live_feature_values)):
        diff = abs(hist_val - live_val)
        match = "✓" if np.isclose(hist_val, live_val, rtol=1e-6, atol=1e-8) else "✗"
        
        if i < 5 or i >= len(hist_feature_values) - 5:  # Show first and last 5
            print(f"{feature_cols[i]:<20} {hist_val:<15.8f} {live_val:<15.8f} {diff:<15.8e} {match}")
        
        if not np.isclose(hist_val, live_val, rtol=1e-6, atol=1e-8):
            mismatches.append((i, feature_cols[i], hist_val, live_val, diff))
    
    print(f"\nTotal features: {len(hist_feature_values)}")
    print(f"Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n❌ Mismatches found:")
        for i, col, hist, live, diff in mismatches[:10]:
            print(f"  {col}: hist={hist:.8f}, live={live:.8f}, diff={diff:.8e}")
        pytest.fail(f"Found {len(mismatches)} mismatches between historical and live features")
    else:
        print("\n✅ All features match perfectly!")


def test_live_vs_historical_features_1hour():
    """Test that live features match historical features for 1-hour bars."""
    ticker = "btcusd"
    
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=24,  # 24 hours of 1-hour bars
        target_bars=24,
        interval="1h",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 2] 1-hour bars: Live vs Historical Feature Consistency")
    print("="*80)
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)
    
    # Get historical features
    df_historical = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with complete features
    feature_cols = [col for col in df_historical.columns if col.startswith('feature_')]
    df_historical = df_historical.dropna(subset=feature_cols)
    
    if len(df_historical) == 0:
        pytest.skip("No complete historical features available")
    
    # Get a recent timestamp
    historical_timestamps = df_historical.xs(ticker, level='ticker').index
    test_timestamp = historical_timestamps[-20]  # 20 bars back from latest
    
    print(f"\nTest timestamp: {test_timestamp}")
    
    # Get historical features
    hist_features = df_historical.xs((ticker, test_timestamp))
    hist_feature_values = hist_features[feature_cols].values
    
    # Get live features for same timestamp
    raw_data = workflow.load_raw(start=start, end=test_timestamp + timedelta(hours=1))
    ticker_data = raw_data.xs(ticker, level='symbol')
    
    resampled = workflow.create_interval_bars(
        ticker_data.reset_index().set_index('open_time'),
        live_mode=False
    )
    
    live_features = workflow.extract_features(
        resampled,
        workflow.number_of_input_bars,
        [test_timestamp]
    )
    
    if live_features.empty:
        pytest.fail("No live features extracted")
    
    live_feature_values = live_features.iloc[0].values
    
    # Compare
    mismatches = []
    for i, (hist_val, live_val) in enumerate(zip(hist_feature_values, live_feature_values)):
        if not np.isclose(hist_val, live_val, rtol=1e-6, atol=1e-8):
            mismatches.append((i, feature_cols[i], hist_val, live_val))
    
    print(f"Total features: {len(hist_feature_values)}")
    print(f"Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n❌ Mismatches found:")
        for i, col, hist, live in mismatches[:5]:
            print(f"  {col}: hist={hist:.8f}, live={live:.8f}")
        pytest.fail(f"Found {len(mismatches)} mismatches")
    else:
        print("✅ All features match perfectly!")


def test_live_features_multi_asset():
    """Test live vs historical features across multiple assets."""
    tickers = ["btcusd", "ethusd"]
    
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=12,  # 1 hour of 5-min bars
        target_bars=12,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 3] Multi-asset: Live vs Historical Feature Consistency")
    print("="*80)
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=2)
    workflow.backfill(start=start)
    
    # Get historical features
    df_historical = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with complete features
    feature_cols = [col for col in df_historical.columns if col.startswith('feature_')]
    df_historical = df_historical.dropna(subset=feature_cols)
    
    raw_data = workflow.load_raw(start=start)
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        print("-"*40)
        
        if ticker not in df_historical.index.get_level_values('ticker'):
            print(f"  Skipping {ticker} - no data")
            continue
        
        # Get a recent timestamp for this ticker
        ticker_hist = df_historical.xs(ticker, level='ticker')
        if len(ticker_hist) == 0:
            print(f"  Skipping {ticker} - no complete features")
            continue
        
        test_timestamp = ticker_hist.index[-5]
        
        # Get historical features
        hist_features = ticker_hist.loc[test_timestamp]
        hist_feature_values = hist_features[feature_cols].values
        
        # Get live features
        ticker_data = raw_data.xs(ticker, level='symbol')
        resampled = workflow.create_interval_bars(
            ticker_data.reset_index().set_index('open_time'),
            live_mode=False
        )
        
        live_features = workflow.extract_rolling_daily_features(
            resampled,
            workflow.number_of_input_bars,
            [test_timestamp]
        )
        
        if live_features.empty:
            print(f"  ❌ No live features extracted for {ticker}")
            continue
        
        live_feature_values = live_features.iloc[0].values
        
        # Compare
        all_match = np.allclose(hist_feature_values, live_feature_values, rtol=1e-6, atol=1e-8)
        max_diff = np.max(np.abs(hist_feature_values - live_feature_values))
        
        print(f"  Timestamp: {test_timestamp}")
        print(f"  Features: {len(hist_feature_values)}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Result: {'✅ Match' if all_match else '❌ Mismatch'}")
        
        if not all_match:
            pytest.fail(f"Features don't match for {ticker}")


if __name__ == "__main__":
    print("="*80)
    print("Live vs Historical Feature Consistency Test Suite")
    print("="*80)
    
    test_live_vs_historical_features_5min()
    test_live_vs_historical_features_1hour()
    test_live_features_multi_asset()
    
    print("\n" + "="*80)
    print("✅ All live vs historical feature tests passed!")
    print("="*80)

