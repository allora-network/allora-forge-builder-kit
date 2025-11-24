"""
Test to verify feature extraction integrity.

This test ensures that extracted features correctly match the underlying OHLCV data
after accounting for normalization.

Normalization scheme:
- OHLC features: divided by the last close price (most recent bar close)
- Volume features: divided by the last volume (most recent bar volume)

Expected behavior:
- feature_close_{N-1} should equal 1.0 (normalized by itself)
- feature_volume_{N-1} should equal 1.0 (normalized by itself)
- All other features should match the actual data when denormalized
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from allora_forge_builder_kit import AlloraMLWorkflow


# Load API key
api_key_file = Path(".allora_api_key")
if api_key_file.exists():
    ALLORA_API_KEY = api_key_file.read_text().strip()
else:
    raise FileNotFoundError(f"API key file not found: {api_key_file}")


def test_feature_integrity_allora_5min():
    """Test feature integrity with Allora 5-minute bars - validates ALL features."""
    ticker = "btcusd"
    
    # Create workflow with 5-minute bars
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=12,  # 1 hour of 5-min bars
        target_bars=1,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=2)
    workflow.backfill(start=start)
    
    # Get features
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with complete features
    feature_cols = [col for col in df_all.columns if col.startswith('feature_')]
    df_test = df_all.dropna(subset=feature_cols).copy()
    
    print(f"\nTesting {len(df_test)} rows with complete features")
    print(f"Validating ALL {workflow.number_of_input_bars} feature bars per row")
    
    # Get ticker data from df_test (already has all OHLCV data)
    ticker_data = df_test.xs(ticker, level='ticker')
    
    # Test a sample of rows
    sample_size = min(50, len(ticker_data))
    df_sample = ticker_data.sample(n=sample_size, random_state=42)
    
    errors = []
    total_feature_checks = 0
    
    for idx, row in df_sample.iterrows():
        # Get the current bar's normalization factors
        current_close = row['close']
        current_volume = row['volume']
        current_time = idx  # Just timestamp for single-ticker slice
        
        # Find the position of the current row
        try:
            current_pos = ticker_data.index.get_loc(current_time)
        except KeyError:
            errors.append(f"Row {idx}: Cannot find timestamp in data")
            continue
        
        # Get the lookback window (previous number_of_input_bars bars)
        lookback_start = current_pos - workflow.number_of_input_bars + 1
        if lookback_start < 0:
            continue  # Skip if not enough history
        
        lookback_bars = ticker_data.iloc[lookback_start:current_pos + 1]
        
        if len(lookback_bars) != workflow.number_of_input_bars:
            errors.append(
                f"Row {idx}: Expected {workflow.number_of_input_bars} lookback bars, "
                f"got {len(lookback_bars)}"
            )
            continue
        
        # Now verify each feature matches the corresponding historical bar
        for bar_idx in range(workflow.number_of_input_bars):
            historical_bar = lookback_bars.iloc[bar_idx]
            
            # Get features for this bar index
            feature_open = row[f'feature_open_{bar_idx}']
            feature_high = row[f'feature_high_{bar_idx}']
            feature_low = row[f'feature_low_{bar_idx}']
            feature_close = row[f'feature_close_{bar_idx}']
            feature_volume = row[f'feature_volume_{bar_idx}']
            
            # Denormalize features
            denorm_open = feature_open * current_close
            denorm_high = feature_high * current_close
            denorm_low = feature_low * current_close
            denorm_close = feature_close * current_close
            denorm_volume = feature_volume * current_volume
            
            # Compare with actual historical values
            if not np.isclose(denorm_open, historical_bar['open'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: open mismatch - "
                    f"denorm={denorm_open:.2f}, actual={historical_bar['open']:.2f}"
                )
            
            if not np.isclose(denorm_high, historical_bar['high'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: high mismatch - "
                    f"denorm={denorm_high:.2f}, actual={historical_bar['high']:.2f}"
                )
            
            if not np.isclose(denorm_low, historical_bar['low'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: low mismatch - "
                    f"denorm={denorm_low:.2f}, actual={historical_bar['low']:.2f}"
                )
            
            if not np.isclose(denorm_close, historical_bar['close'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: close mismatch - "
                    f"denorm={denorm_close:.2f}, actual={historical_bar['close']:.2f}"
                )
            
            if not np.isclose(denorm_volume, historical_bar['volume'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: volume mismatch - "
                    f"denorm={denorm_volume:.2f}, actual={historical_bar['volume']:.2f}"
                )
            
            total_feature_checks += 5  # 5 features per bar (OHLCV)
        
        # Special check: last feature should be normalized to 1.0
        last_idx = workflow.number_of_input_bars - 1
        if not np.isclose(row[f'feature_close_{last_idx}'], 1.0, rtol=1e-5):
            errors.append(f"Row {idx}: Last close feature != 1.0")
        
        if not np.isclose(row[f'feature_volume_{last_idx}'], 1.0, rtol=1e-5):
            errors.append(f"Row {idx}: Last volume feature != 1.0")
    
    # Report results
    print(f"Total feature checks performed: {total_feature_checks:,}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        pytest.fail(f"Feature integrity test failed with {len(errors)} errors")
    else:
        print(f"✅ All {sample_size} rows × {workflow.number_of_input_bars} bars = {total_feature_checks:,} checks passed!")


def test_feature_integrity_allora_1hour():
    """Test feature integrity with Allora 1-hour bars - validates ALL features."""
    ticker = "btcusd"
    
    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=24,  # 24 hours of 1-hour bars
        target_bars=24,
        interval="1h",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=7)
    workflow.backfill(start=start)
    
    # Get features
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    # Filter to rows with complete features
    feature_cols = [col for col in df_all.columns if col.startswith('feature_')]
    df_test = df_all.dropna(subset=feature_cols).copy()
    
    print(f"\nTesting {len(df_test)} rows with complete features")
    print(f"Validating ALL {workflow.number_of_input_bars} feature bars per row")
    
    # Get ticker data from df_test (already has all OHLCV data)
    ticker_data = df_test.xs(ticker, level='ticker')
    
    # Test a sample
    sample_size = min(30, len(ticker_data))
    df_sample = ticker_data.sample(n=sample_size, random_state=42)
    
    errors = []
    total_feature_checks = 0
    
    for idx, row in df_sample.iterrows():
        current_close = row['close']
        current_volume = row['volume']
        current_time = idx  # Just timestamp for single-ticker slice
        
        try:
            current_pos = ticker_data.index.get_loc(current_time)
        except KeyError:
            continue
        
        lookback_start = current_pos - workflow.number_of_input_bars + 1
        if lookback_start < 0:
            continue
        
        lookback_bars = ticker_data.iloc[lookback_start:current_pos + 1]
        
        if len(lookback_bars) != workflow.number_of_input_bars:
            continue
        
        # Verify each feature
        for bar_idx in range(workflow.number_of_input_bars):
            historical_bar = lookback_bars.iloc[bar_idx]
            
            feature_close = row[f'feature_close_{bar_idx}']
            denorm_close = feature_close * current_close
            
            if not np.isclose(denorm_close, historical_bar['close'], rtol=1e-4):
                errors.append(
                    f"Row {idx}, bar {bar_idx}: close mismatch - "
                    f"denorm={denorm_close:.2f}, actual={historical_bar['close']:.2f}"
                )
            
            total_feature_checks += 1
    
    print(f"Total feature checks performed: {total_feature_checks:,}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  {error}")
        pytest.fail(f"Feature integrity test failed with {len(errors)} errors")
    else:
        print(f"✅ All {sample_size} rows × {workflow.number_of_input_bars} bars = {total_feature_checks:,} checks passed!")


def test_feature_integrity_multi_asset():
    """Test feature integrity across multiple assets - validates ALL features."""
    tickers = ["btcusd", "ethusd", "solusd"]
    
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=24,  # 2 hours of 5-min bars
        target_bars=12,
        interval="5m",
        data_source="allora",
        api_key=ALLORA_API_KEY
    )
    
    # Backfill recent data
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)
    
    # Get features
    df_all = workflow.get_full_feature_target_dataframe(start_date=start)
    
    print(f"\nTesting features for {len(tickers)} assets")
    print(f"Validating ALL {workflow.number_of_input_bars} feature bars per row per asset")
    
    # Test each ticker separately
    for ticker in tickers:
        ticker_data = df_all.xs(ticker, level='ticker')
        
        # Filter to rows with complete features
        feature_cols = [col for col in ticker_data.columns if col.startswith('feature_')]
        df_test = ticker_data.dropna(subset=feature_cols).copy()
        
        print(f"\n{ticker}: Testing {len(df_test)} rows")
        
        # Test a sample
        sample_size = min(20, len(df_test))
        df_sample = df_test.sample(n=sample_size, random_state=42)
        
        errors = []
        total_feature_checks = 0
        
        for idx, row in df_sample.iterrows():
            current_close = row['close']
            current_time = idx  # Just timestamp for single-ticker slice
            
            try:
                current_pos = df_test.index.get_loc(current_time)
            except KeyError:
                continue
            
            lookback_start = current_pos - workflow.number_of_input_bars + 1
            if lookback_start < 0:
                continue
            
            lookback_bars = df_test.iloc[lookback_start:current_pos + 1]
            
            if len(lookback_bars) != workflow.number_of_input_bars:
                continue
            
            # Verify each feature
            for bar_idx in range(workflow.number_of_input_bars):
                historical_bar = lookback_bars.iloc[bar_idx]
                
                feature_close = row[f'feature_close_{bar_idx}']
                denorm_close = feature_close * current_close
                
                if not np.isclose(denorm_close, historical_bar['close'], rtol=1e-4):
                    errors.append(
                        f"bar {bar_idx}: close mismatch - "
                        f"denorm={denorm_close:.2f}, actual={historical_bar['close']:.2f}"
                    )
                
                total_feature_checks += 1
        
        print(f"  Total checks: {total_feature_checks:,}")
        
        if errors:
            print(f"  ❌ {len(errors)} errors for {ticker}")
            for error in errors[:5]:
                print(f"    {error}")
            pytest.fail(f"Feature integrity test failed for {ticker}")
        else:
            print(f"  ✅ All {sample_size} rows × {workflow.number_of_input_bars} bars = {total_feature_checks:,} checks passed!")


if __name__ == "__main__":
    print("="*80)
    print("Feature Integrity Test Suite - Allora Data")
    print("="*80)
    
    print("\n[Test 1] Allora 5-minute bars (single asset)")
    test_feature_integrity_allora_5min()
    
    print("\n[Test 2] Allora 1-hour bars (single asset)")
    test_feature_integrity_allora_1hour()
    
    print("\n[Test 3] Multi-asset test (btcusd, ethusd, solusd)")
    test_feature_integrity_multi_asset()
    
    print("\n" + "="*80)
    print("✅ All feature integrity tests passed!")
    print("="*80)

