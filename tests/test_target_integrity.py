"""
Test to verify target values are correctly computed.

This ensures that the target (log return) corresponds exactly to the actual
future price movement by finding the correct future bar and recalculating.
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
            return p.read_text().strip()
    return None


ALLORA_API_KEY = _load_api_key()
pytestmark = pytest.mark.skipif(
    ALLORA_API_KEY is None, reason="ALLORA_API_KEY not available"
)


def test_target_integrity_5min():
    """Test that targets match manually calculated log returns for 5-min bars."""
    ticker = "btcusd"
    target_bars_ahead = 12  # Predict 12 bars (1 hour) ahead
    
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=12,
        target_bars=target_bars_ahead,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 1] 5-minute bars: Target Integrity Check")
    print("="*80)
    print(f"Target: {target_bars_ahead} bars ahead")
    
    # Backfill data
    start = datetime.now(timezone.utc) - timedelta(days=2)
    workflow.backfill(start=start)
    
    # Get features and targets
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with targets
    df_test = df_all.dropna(subset=['target']).copy()
    
    if len(df_test) == 0:
        pytest.skip("No complete targets available")
    
    print(f"\nTesting {len(df_test)} rows with targets")
    
    # Get ticker data from df_test (already has all OHLCV data)
    ticker_data = df_test.xs(ticker, level='ticker')
    
    # Sample rows to test
    sample_size = min(100, len(df_test))
    df_sample = ticker_data.sample(n=sample_size, random_state=42)
    
    errors = []
    
    for idx, row in df_sample.iterrows():
        # idx is just the timestamp for single-ticker slice
        current_time = idx
        current_close = row['close']
        computed_target = row['target']
        
        # Find current position
        try:
            current_pos = ticker_data.index.get_loc(current_time)
        except KeyError:
            errors.append(f"Row {idx}: Cannot find timestamp in data")
            continue
        
        # Find future bar
        future_pos = current_pos + target_bars_ahead
        if future_pos >= len(ticker_data):
            # Expected - no future data for most recent bars
            continue
        
        future_close = ticker_data.iloc[future_pos]['close']
        
        # Manually calculate log return
        manual_target = np.log(future_close / current_close)
        
        # Compare
        if not np.isclose(computed_target, manual_target, rtol=1e-6, atol=1e-8):
            errors.append(
                f"Row {idx}: Target mismatch - "
                f"computed={computed_target:.8f}, "
                f"manual={manual_target:.8f}, "
                f"diff={abs(computed_target - manual_target):.8e}"
            )
    
    print(f"Rows tested: {len(df_sample)}")
    print(f"Errors found: {len(errors)}")
    
    if errors:
        print("\n❌ Target mismatches found:")
        for error in errors[:10]:
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        pytest.fail(f"Found {len(errors)} target mismatches")
    else:
        print("✅ All targets match manually calculated log returns!")


def test_target_integrity_1hour():
    """Test that targets match manually calculated log returns for 1-hour bars."""
    ticker = "btcusd"
    target_bars_ahead = 24  # Predict 24 bars (24 hours) ahead
    
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=24,
        target_bars=target_bars_ahead,
        interval="1h",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 2] 1-hour bars: Target Integrity Check")
    print("="*80)
    print(f"Target: {target_bars_ahead} bars ahead")
    
    # Backfill data
    start = datetime.now(timezone.utc) - timedelta(days=7)
    workflow.backfill(start=start)
    
    # Get features and targets
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with targets
    df_test = df_all.dropna(subset=['target']).copy()
    
    if len(df_test) == 0:
        pytest.skip("No complete targets available")
    
    print(f"\nTesting {len(df_test)} rows with targets")
    
    # Get ticker data from df_test (already has all OHLCV data)
    ticker_data = df_test.xs(ticker, level='ticker')
    
    # Sample rows
    sample_size = min(50, len(df_test))
    df_sample = ticker_data.sample(n=sample_size, random_state=42)
    
    errors = []
    
    for idx, row in df_sample.iterrows():
        current_time = idx
        current_close = row['close']
        computed_target = row['target']
        
        try:
            current_pos = ticker_data.index.get_loc(current_time)
        except KeyError:
            continue
        
        future_pos = current_pos + target_bars_ahead
        if future_pos >= len(ticker_data):
            continue
        
        future_close = ticker_data.iloc[future_pos]['close']
        manual_target = np.log(future_close / current_close)
        
        if not np.isclose(computed_target, manual_target, rtol=1e-6, atol=1e-8):
            errors.append(
                f"Row {idx}: computed={computed_target:.8f}, "
                f"manual={manual_target:.8f}"
            )
    
    print(f"Rows tested: {len(df_sample)}")
    print(f"Errors found: {len(errors)}")
    
    if errors:
        print("\n❌ Target mismatches found:")
        for error in errors[:5]:
            print(f"  {error}")
        pytest.fail(f"Found {len(errors)} target mismatches")
    else:
        print("✅ All targets match manually calculated log returns!")


def test_target_integrity_multi_asset():
    """Test target integrity across multiple assets."""
    tickers = ["btcusd", "ethusd"]
    target_bars_ahead = 24  # 2 hours ahead with 5-min bars
    
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=24,
        target_bars=target_bars_ahead,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    print("\n[Test 3] Multi-asset: Target Integrity Check")
    print("="*80)
    print(f"Target: {target_bars_ahead} bars ahead")
    
    # Backfill data
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)
    
    # Get features and targets
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        print("-"*40)
        
        if ticker not in df_all.index.get_level_values('ticker'):
            print(f"  Skipping {ticker} - no data")
            continue
        
        # Get ticker data from df_all (already has all OHLCV data)
        ticker_data = df_all.xs(ticker, level='ticker')
        df_test = ticker_data.dropna(subset=['target']).copy()
        
        if len(df_test) == 0:
            print(f"  Skipping {ticker} - no complete targets")
            continue
        
        sample_size = min(30, len(df_test))
        df_sample = df_test.sample(n=sample_size, random_state=42)
        
        errors = []
        
        for idx, row in df_sample.iterrows():
            current_time = idx
            current_close = row['close']
            computed_target = row['target']
            
            try:
                current_pos = df_test.index.get_loc(current_time)
            except KeyError:
                continue
            
            future_pos = current_pos + target_bars_ahead
            if future_pos >= len(df_test):
                continue
            
            future_close = df_test.iloc[future_pos]['close']
            manual_target = np.log(future_close / current_close)
            
            if not np.isclose(computed_target, manual_target, rtol=1e-6, atol=1e-8):
                errors.append(f"Mismatch at {current_time}")
        
        print(f"  Rows tested: {len(df_sample)}")
        print(f"  Errors: {len(errors)}")
        
        if errors:
            print(f"  ❌ {len(errors)} target mismatches for {ticker}")
            pytest.fail(f"Target integrity test failed for {ticker}")
        else:
            print(f"  ✅ All targets correct for {ticker}")


if __name__ == "__main__":
    print("="*80)
    print("Target Integrity Test Suite")
    print("="*80)
    
    test_target_integrity_5min()
    test_target_integrity_1hour()
    test_target_integrity_multi_asset()
    
    print("\n" + "="*80)
    print("✅ All target integrity tests passed!")
    print("="*80)

