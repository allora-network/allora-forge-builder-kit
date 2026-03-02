"""
Comprehensive tests for DataManager implementations and Workflow integration.

Run with:
    pytest tests/test_data_managers.py -v

For integration tests (requires API keys and network):
    RUN_INTEGRATION_TESTS=1 ALLORA_API_KEY=your-key pytest tests/test_data_managers.py -v
"""

import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest
import pandas as pd
import polars as pl

from allora_forge_builder_kit import (
    DataManager,
    BinanceDataManager,
    AtlasDataManager,
    BaseDataManager,
    AlloraMLWorkflow,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_symbols_binance():
    """Binance test symbols (small set for speed)."""
    return ["BTCUSDT"]


@pytest.fixture
def test_symbols_allora():
    """Allora test symbols (small set for speed).
    
    Allora uses lowercase format: 'btcusd', 'ethusd', 'solusd'
    The data manager will also normalize "BTC/USD" to "btcusd" internally.
    """
    return ["btcusd"]


@pytest.fixture
def allora_api_key():
    """Get Allora API key from environment."""
    key = os.environ.get("ALLORA_API_KEY")
    if not key:
        pytest.skip("ALLORA_API_KEY not set")
    return key


@pytest.fixture
def integration_check():
    """Check if integration tests should run."""
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Integration test. Set RUN_INTEGRATION_TESTS=1 to enable.")


# ============================================================================
# Unit Tests - Factory Pattern
# ============================================================================

def test_factory_returns_binance_manager(tmp_path):
    """Test that factory returns BinanceDataManager for source='binance'."""
    dm = DataManager(
        source="binance",
        interval="5m",
        market="futures",
        base_dir=str(tmp_path / "binance_data")
    )
    
    assert isinstance(dm, BinanceDataManager)
    assert isinstance(dm, BaseDataManager)
    assert dm.interval == "5m"
    assert dm.market == "futures"


def test_factory_returns_allora_manager(tmp_path):
    """Test that factory returns AtlasDataManager for source='allora' (new default)."""
    from allora_forge_builder_kit.atlas_data_manager import AtlasDataManager

    dm = DataManager(
        source="allora",
        interval="5m",
        api_key="test-key",
        base_dir=str(tmp_path / "allora_data")
    )
    
    assert isinstance(dm, AtlasDataManager)
    assert isinstance(dm, BaseDataManager)
    assert dm.interval == "5m"
    assert dm.api_key == "test-key"


def test_factory_invalid_source():
    """Test that factory raises error for invalid source."""
    with pytest.raises(ValueError, match="Unknown data source"):
        DataManager(source="invalid")


def test_factory_allora_missing_api_key():
    """Test that factory requires api_key for Allora."""
    with pytest.raises(ValueError, match="api_key is required"):
        DataManager(source="allora", interval="5m")


# ============================================================================
# Unit Tests - BinanceDataManager
# ============================================================================

def test_binance_manager_initialization(tmp_path):
    """Test BinanceDataManager initialization."""
    dm = BinanceDataManager(
        interval="5m",
        market="futures",
        symbols=["BTCUSDT"],
        base_dir=str(tmp_path / "binance_data")
    )
    
    assert dm.interval == "5m"
    assert dm.market == "futures"
    assert "BTCUSDT" in dm.symbols
    assert dm.base_dir == str(tmp_path / "binance_data")


def test_binance_manager_default_directory():
    """Test BinanceDataManager uses source-specific default directory."""
    dm = BinanceDataManager(interval="5m")
    assert dm.base_dir == "parquet_data_binance"


def test_binance_manager_parse_kline():
    """Test Binance kline parsing to standardized format."""
    dm = BinanceDataManager(interval="5m")
    
    # Binance kline format
    kline = [
        1633017600000,  # open_time
        "50000.0",      # open
        "50100.0",      # high
        "49900.0",      # low
        "50050.0",      # close
        "100.5",        # volume
        1633017899999,  # close_time
        "5000000.0",    # quote_volume
        1000,           # n_trades
        "50.0",         # taker_base_vol
        "2500000.0",    # taker_quote_vol
        "0"             # ignore
    ]
    
    bar = dm._parse_binance_kline(kline, "BTCUSDT")
    
    assert bar["symbol"] == "BTCUSDT"
    assert bar["open"] == 50000.0
    assert bar["high"] == 50100.0
    assert bar["low"] == 49900.0
    assert bar["close"] == 50050.0
    assert bar["volume"] == 100.5
    assert bar["quote_volume"] == 5000000.0
    assert bar["n_trades"] == 1000
    assert "open_time" in bar
    assert isinstance(bar["open_time"], datetime)


# ============================================================================
# Unit Tests - AtlasDataManager
# ============================================================================

def test_atlas_manager_initialization(tmp_path):
    """Test AtlasDataManager initialization."""
    dm = AtlasDataManager(
        api_key="test-key",
        interval="5m",
        symbols=["btcusd"],
        base_dir=str(tmp_path / "allora_data")
    )

    assert dm.interval == "5m"
    assert dm.api_key == "test-key"
    assert dm.base_dir == str(tmp_path / "allora_data")


def test_atlas_manager_default_directory():
    """Test AtlasDataManager uses source-specific default directory."""
    dm = AtlasDataManager(api_key="test-key", interval="5m")
    assert dm.base_dir == "parquet_data_allora"


# ============================================================================
# Unit Tests - Storage Structure
# ============================================================================

def test_storage_separation(tmp_path):
    """Test that different sources use different directories."""
    binance_dm = BinanceDataManager(base_dir=str(tmp_path / "binance"))
    atlas_dm = AtlasDataManager(api_key="test", base_dir=str(tmp_path / "allora"))

    assert binance_dm.base_dir != atlas_dm.base_dir
    assert "binance" in binance_dm.base_dir
    assert "allora" in atlas_dm.base_dir


def test_partition_path(tmp_path):
    """Test partition path generation."""
    dm = BinanceDataManager(base_dir=str(tmp_path / "data"))
    
    path = dm._partition_path("BTCUSDT", "2025-10-06")
    
    assert "symbol=BTCUSDT" in path
    assert "dt=2025-10-06.parquet" in path


def test_rows_to_dataframe_nested_format():
    """Test _rows_to_dataframe with nested /api/rows/ format."""
    from allora_forge_builder_kit.atlas_data_manager import AtlasDataManager

    rows = [
        {"timestamp": "2024-06-01T00:00:00Z", "values": {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 50}},
        {"timestamp": "2024-06-01T00:01:00Z", "values": {"open": 105.0, "high": 115.0, "low": 95.0, "close": 108.0, "volume": 60}},
    ]
    df = AtlasDataManager._rows_to_dataframe(rows)
    assert len(df) == 2
    assert df["open"].iloc[0] == 100.0
    assert df["close"].iloc[1] == 108.0
    assert not df[["open", "high", "low", "close"]].isna().any().any()


def test_rows_to_dataframe_flat_format():
    """Test _rows_to_dataframe with flat bulk_download format."""
    from allora_forge_builder_kit.atlas_data_manager import AtlasDataManager

    rows = [
        {"timestamp": "2024-06-01T00:00:00Z", "open": 67475.15, "high": 67515.47, "low": 67400.0, "close": 67506.88, "volume": 123.4},
        {"timestamp": "2024-06-01T00:01:00Z", "open": 67506.88, "high": 67550.0, "low": 67490.0, "close": 67530.0, "volume": 456.7},
    ]
    df = AtlasDataManager._rows_to_dataframe(rows)
    assert len(df) == 2
    assert df["open"].iloc[0] == 67475.15
    assert df["close"].iloc[1] == 67530.0
    assert not df[["open", "high", "low", "close"]].isna().any().any()


# ============================================================================
# Unit Tests - Workflow Integration
# ============================================================================

def test_workflow_with_binance_string_api(tmp_path):
    """Test AlloraMLWorkflow with Binance using string API."""
    workflow = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=24,
        interval="5m",
        data_source="binance",
        market="futures",
        base_dir=str(tmp_path / "binance_data")
    )
    
    assert isinstance(workflow._dm, BinanceDataManager)
    assert workflow.interval == "5m"


def test_workflow_with_allora_string_api(tmp_path):
    """Test AlloraMLWorkflow with Allora using string API (now Atlas backend)."""
    from allora_forge_builder_kit.atlas_data_manager import AtlasDataManager

    workflow = AlloraMLWorkflow(
        tickers=["btcusd"],
        number_of_input_bars=288,
        target_bars=24,
        interval="5m",
        data_source="allora",
        api_key="test-key",
        base_dir=str(tmp_path / "allora_data")
    )
    
    assert isinstance(workflow._dm, AtlasDataManager)
    assert workflow.interval == "5m"


def test_workflow_with_explicit_manager(tmp_path):
    """Test AlloraMLWorkflow with explicit data manager."""
    dm = BinanceDataManager(
        interval="5m",
        market="futures",
        base_dir=str(tmp_path / "data")
    )
    
    workflow = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=24,
        data_manager=dm
    )
    
    assert workflow._dm is dm
    assert isinstance(workflow._dm, BinanceDataManager)


def test_workflow_invalid_manager():
    """Test that workflow rejects non-BaseDataManager instances."""
    with pytest.raises(TypeError, match="must be an instance of BaseDataManager"):
        AlloraMLWorkflow(
            tickers=["BTCUSDT"],
            number_of_input_bars=288,
            target_bars=24,
            data_manager="not-a-manager"  # Should be BaseDataManager instance
        )


# ============================================================================
# Integration Tests - BinanceDataManager (requires network)
# ============================================================================

@pytest.mark.integration
def test_binance_backfill_and_load(tmp_path, test_symbols_binance, integration_check):
    """Test Binance backfill and load operations."""
    dm = BinanceDataManager(
        interval="5m",
        market="futures",
        symbols=test_symbols_binance,
        base_dir=str(tmp_path / "binance_data")
    )
    
    # Backfill last 2 days
    start = datetime.now(timezone.utc) - timedelta(days=2)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    
    dm.backfill_symbol(test_symbols_binance[0], start, end)
    
    # Load data
    df = dm.load_pandas(test_symbols_binance, start=start, end=end)
    
    assert not df.empty
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["symbol", "open_time"]
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "n_trades" in df.columns


@pytest.mark.integration
def test_binance_live_snapshot(test_symbols_binance, integration_check):
    """Test Binance get_live_snapshot."""
    dm = BinanceDataManager(interval="5m", market="futures")
    
    snapshot = dm.get_live_snapshot(test_symbols_binance)
    
    assert not snapshot.empty
    assert isinstance(snapshot.index, pd.MultiIndex)
    assert len(snapshot) == len(test_symbols_binance)
    assert "open" in snapshot.columns
    assert "close" in snapshot.columns
    assert "volume" in snapshot.columns
    assert "n_trades" in snapshot.columns


@pytest.mark.integration
def test_binance_get_live_1min_data(test_symbols_binance, integration_check):
    """Test Binance get_live_1min_data returns 1-minute bars."""
    dm = BinanceDataManager(interval="5m", market="futures")
    
    # Fetch 2 hours of 1-minute data
    df_1min = dm.get_live_1min_data(test_symbols_binance[0], hours_back=2)
    
    assert not df_1min.empty
    assert isinstance(df_1min.index, pd.DatetimeIndex)
    assert list(df_1min.columns) == ["open", "high", "low", "close", "volume"]
    
    # Verify it's 1-minute bars
    if len(df_1min) > 1:
        time_diff = (df_1min.index[1] - df_1min.index[0]).total_seconds() / 60
        assert abs(time_diff - 1.0) < 0.1, f"Expected 1-minute bars, got {time_diff} minutes"
    
    # Verify we got roughly the right amount of data (should be ~120 bars for 2 hours)
    assert len(df_1min) >= 100, "Should have at least 100 1-minute bars in 2 hours"


# ============================================================================
# Integration Tests - AtlasDataManager (requires API key + network)
# Note: Atlas data may lag wall-clock time, so bar count thresholds are
# relaxed compared to Binance (which streams in real time).
# ============================================================================

@pytest.mark.integration
def test_atlas_backfill_and_load(tmp_path, test_symbols_allora, allora_api_key, integration_check):
    """Test Atlas backfill and load operations."""
    dm = AtlasDataManager(
        api_key=allora_api_key,
        interval="5m",
        symbols=test_symbols_allora,
        base_dir=str(tmp_path / "allora_data")
    )

    start = datetime.now(timezone.utc) - timedelta(days=3)

    dm.backfill_symbol(test_symbols_allora[0], start, datetime.now(timezone.utc))

    df = dm.load_pandas(test_symbols_allora, start=start)

    if df.empty:
        print(f"[test] Atlas API returned no data for {test_symbols_allora[0]}")
        return

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["symbol", "open_time"]
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns


@pytest.mark.integration
def test_atlas_live_snapshot(test_symbols_allora, allora_api_key, integration_check):
    """Test Atlas get_live_snapshot."""
    dm = AtlasDataManager(api_key=allora_api_key, interval="5m")

    snapshot = dm.get_live_snapshot(test_symbols_allora)

    assert not snapshot.empty
    assert isinstance(snapshot.index, pd.MultiIndex)
    assert len(snapshot) == len(test_symbols_allora)
    assert "open" in snapshot.columns
    assert "close" in snapshot.columns
    assert "volume" in snapshot.columns


@pytest.mark.integration
def test_atlas_get_live_1min_data(test_symbols_allora, allora_api_key, integration_check):
    """Test Atlas get_live_1min_data returns 1-minute bars."""
    dm = AtlasDataManager(api_key=allora_api_key, interval="5m")

    df_1min = dm.get_live_1min_data(test_symbols_allora[0], hours_back=2)

    if df_1min.empty:
        pytest.skip("Atlas API returned no data")

    assert isinstance(df_1min.index, pd.DatetimeIndex)
    assert list(df_1min.columns) == ["open", "high", "low", "close", "volume"]

    if len(df_1min) > 1:
        time_diff = (df_1min.index[1] - df_1min.index[0]).total_seconds() / 60
        assert abs(time_diff - 1.0) < 0.1, f"Expected 1-minute bars, got {time_diff} minutes"

    assert len(df_1min) >= 30, (
        f"Should have at least 30 1-minute bars (got {len(df_1min)}); "
        f"Atlas data may lag wall-clock time"
    )


@pytest.mark.integration
def test_atlas_live_window_coverage_btcusd(allora_api_key, integration_check):
    """Integration check: Atlas BTC/USD live window has expected recency and row coverage."""
    dm = AtlasDataManager(api_key=allora_api_key, interval="1m")

    hours_back = 6
    df_1min = dm.get_live_1min_data("btcusd", hours_back=hours_back)

    if df_1min.empty:
        pytest.skip("Atlas API returned no data for btcusd")

    # Structure checks
    assert isinstance(df_1min.index, pd.DatetimeIndex)
    assert df_1min.index.tz is not None, "Expected timezone-aware UTC index"
    assert list(df_1min.columns) == ["open", "high", "low", "close", "volume"]

    # Monotonic + roughly 1-minute spacing
    assert df_1min.index.is_monotonic_increasing
    if len(df_1min) > 1:
        minute_diffs = pd.Series(df_1min.index).diff().dropna().dt.total_seconds() / 60.0
        assert (minute_diffs >= 0.9).all() and (minute_diffs <= 1.1).all(), (
            f"Expected ~1-minute spacing, saw min={minute_diffs.min()} max={minute_diffs.max()}"
        )

    # Coverage checks (allowing provider lag / occasional missing minute)
    expected_rows = hours_back * 60
    assert len(df_1min) >= int(expected_rows * 0.85), (
        f"Expected at least 85% of {expected_rows} rows, got {len(df_1min)}"
    )
    assert len(df_1min) <= expected_rows + 5, (
        f"Expected at most {expected_rows + 5} rows, got {len(df_1min)}"
    )

    # Recency: latest row should be close to current UTC minute
    now_utc = datetime.now(timezone.utc)
    latest_ts = df_1min.index[-1].to_pydatetime()
    lag_minutes = (now_utc - latest_ts).total_seconds() / 60.0
    assert lag_minutes <= 15, f"Latest Atlas bar is too old: lag={lag_minutes:.1f} min"


# ============================================================================
# Integration Tests - Workflow with Binance
# ============================================================================

@pytest.mark.integration
def test_workflow_binance_backfill_and_features(tmp_path, test_symbols_binance, integration_check):
    """Test full workflow with Binance: backfill + feature extraction."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=16,
        interval="5m",
        data_source="binance",
        market="futures",
        base_dir=str(tmp_path / "binance_data")
    )
    
    # Backfill last 3 days
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)
    
    # Load raw data
    df_raw = workflow.load_raw(start=start)
    assert not df_raw.empty
    
    # Extract features
    df_features = workflow.get_full_feature_target_dataframe(start_date=start)
    assert not df_features.empty
    assert "target" in df_features.columns
    
    # Check feature columns exist
    feature_cols = [col for col in df_features.columns if col.startswith("feature_")]
    assert len(feature_cols) > 0


@pytest.mark.integration
def test_workflow_binance_get_live_features(test_symbols_binance, integration_check):
    """Test workflow get_live_features with Binance."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        number_of_input_bars=24,  # 2 hours of 5-min bars
        target_bars=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    # Get live features
    features = workflow.get_live_features(test_symbols_binance[0])
    
    assert not features.empty
    assert features.shape[0] == 1  # Should return 1 row (latest bar)
    assert features.shape[1] == 24 * 5  # 24 bars × 5 OHLCV features
    assert isinstance(features.index, pd.DatetimeIndex)
    
    # Verify features are normalized (should be around 1.0)
    assert (features.iloc[0] > 0).all(), "All features should be positive"
    assert (features.iloc[0] < 10).all(), "Features should be normalized (< 10)"


@pytest.mark.integration
def test_workflow_binance_live_features_end_to_end(test_symbols_binance, integration_check):
    """End-to-end test of get_live_features() with Binance validation and visual inspection."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        number_of_input_bars=24,  # 2 hours of 5-min bars
        target_bars=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    print(f"\n{'='*80}")
    print(f"End-to-End: get_live_features() for {test_symbols_binance[0]}")
    print(f"Configuration:")
    print(f"  - interval: {workflow.interval}")
    print(f"  - number_of_input_bars: {workflow.number_of_input_bars}")
    
    # Get live features
    features = workflow.get_live_features(test_symbols_binance[0])
    
    print(f"\nFeatures extracted:")
    print(f"  - Shape: {features.shape}")
    print(f"  - Timestamp: {features.index[0]}")
    print(f"  - First 10 features: {features.iloc[0, :10].values}")
    print(f"  - All positive: {(features.iloc[0] > 0).all()}")
    print(f"  - All normalized (<10): {(features.iloc[0] < 10).all()}")
    print(f"{'='*80}\n")
    
    # Verify shape
    assert features.shape == (1, 120), f"Expected (1, 120), got {features.shape}"
    
    # Verify features are normalized
    assert (features.iloc[0] > 0).all(), "All features should be positive"
    assert (features.iloc[0] < 10).all(), "Features should be normalized (< 10)"
    
    # Verify index is datetime
    assert isinstance(features.index, pd.DatetimeIndex)
    
    # Verify we can use it for prediction (no NaN values)
    assert not features.isnull().any().any(), "Features should not contain NaN"


# ============================================================================
# Integration Tests - Workflow with Allora
# ============================================================================

@pytest.mark.integration
def test_workflow_allora_backfill_and_features(tmp_path, test_symbols_allora, allora_api_key, integration_check):
    """Test full workflow with Allora: backfill + feature extraction."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_allora,
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=16,
        interval="5m",
        data_source="allora",
        api_key=allora_api_key,
        base_dir=str(tmp_path / "allora_data")
    )
    
    # Backfill last 3 days
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)
    
    # Load raw data
    df_raw = workflow.load_raw(start=start)
    assert not df_raw.empty
    
    # Extract features
    df_features = workflow.get_full_feature_target_dataframe(start_date=start)
    assert not df_features.empty
    assert "target" in df_features.columns
    
    # Check feature columns exist
    feature_cols = [col for col in df_features.columns if col.startswith("feature_")]
    assert len(feature_cols) > 0


@pytest.mark.integration
def test_workflow_allora_get_live_features(test_symbols_allora, allora_api_key, integration_check):
    """Test workflow get_live_features with Allora (Atlas backend)."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_allora,
        number_of_input_bars=24,  # 2 hours of 5-min bars
        target_bars=16,
        interval="5m",
        data_source="allora",
        api_key=allora_api_key
    )
    
    # Get live features
    try:
        features = workflow.get_live_features(test_symbols_allora[0])
    except ValueError as e:
        if "No 1-minute data returned" in str(e) or "Not enough historical data" in str(e):
            pytest.skip(f"Allora API has limited data availability: {e}")
        raise
    
    assert not features.empty
    assert features.shape[0] == 1  # Should return 1 row (latest bar)
    assert features.shape[1] == 24 * 5  # 24 bars × 5 OHLCV features
    assert isinstance(features.index, pd.DatetimeIndex)
    
    # Verify features are normalized (should be around 1.0)
    assert (features.iloc[0] > 0).all(), "All features should be positive"
    assert (features.iloc[0] < 10).all(), "Features should be normalized (< 10)"


@pytest.mark.integration
def test_workflow_allora_end_to_end(tmp_path, test_symbols_allora, allora_api_key, integration_check):
    """End-to-end Atlas test: backfill → features → targets → live features all validate."""
    ticker = test_symbols_allora[0]
    n_bars = 24
    target_bars = 12

    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=n_bars,
        target_bars=target_bars,
        interval="5m",
        data_source="allora",
        api_key=allora_api_key,
        base_dir=str(tmp_path / "atlas_e2e"),
    )

    # 1. Backfill
    start = datetime.now(timezone.utc) - timedelta(days=3)
    workflow.backfill(start=start)

    # 2. Historical features + targets
    df = workflow.get_full_feature_target_dataframe(start_date=start)
    assert not df.empty, "Feature dataframe should not be empty"

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    expected_features = n_bars * 5  # OHLCV per bar
    assert len(feature_cols) == expected_features, (
        f"Expected {expected_features} feature columns, got {len(feature_cols)}"
    )
    assert "target" in df.columns

    # Check at least some rows have complete features (not all NaN)
    complete = df.dropna(subset=feature_cols)
    assert len(complete) > 0, "Should have rows with complete features"

    # Targets should be log-returns (small floats centred near 0)
    targets = df["target"].dropna()
    if len(targets) > 0:
        assert targets.abs().max() < 1.0, "Log-return targets should be small"

    # 3. Live features
    try:
        live = workflow.get_live_features(ticker)
    except ValueError as e:
        if "No 1-minute data" in str(e) or "Not enough" in str(e):
            pytest.skip(f"Atlas live data limited: {e}")
        raise

    assert live.shape == (1, expected_features), f"Expected (1, {expected_features}), got {live.shape}"
    assert isinstance(live.index, pd.DatetimeIndex)
    assert not live.isnull().any().any(), "Live features should not contain NaN"
    assert (live.iloc[0] > 0).all(), "All live features should be positive"
    assert (live.iloc[0] < 10).all(), "Live features should be normalized (< 10)"

    # 4. Verify last close feature == 1.0 (self-normalised)
    last_close_col = f"feature_close_{n_bars - 1}"
    assert np.isclose(live.iloc[0][last_close_col], 1.0, rtol=1e-5), (
        f"{last_close_col} should be 1.0 (normalised), got {live.iloc[0][last_close_col]}"
    )


# ============================================================================
# Integration Tests - Compare Sources
# ============================================================================

@pytest.mark.integration
def test_both_sources_coexist(tmp_path, integration_check):
    """Test that Binance and Allora can be used simultaneously without collision."""
    allora_key = os.environ.get("ALLORA_API_KEY", "test-key")
    
    # Create both workflows
    binance_wf = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=16,
        data_source="binance",
        market="futures",
        base_dir=str(tmp_path / "binance")
    )
    
    allora_wf = AlloraMLWorkflow(
        tickers=["btcusd"],  # Allora uses lowercase format
        number_of_input_bars=288,  # 24 hours of 5-min bars
        target_bars=16,
        data_source="allora",
        api_key=allora_key,
        base_dir=str(tmp_path / "allora")
    )
    
    # Verify different data managers
    assert isinstance(binance_wf._dm, BinanceDataManager)
    assert isinstance(allora_wf._dm, AtlasDataManager)
    
    # Verify different storage
    assert binance_wf._dm.base_dir != allora_wf._dm.base_dir
    assert "binance" in binance_wf._dm.base_dir
    assert "allora" in allora_wf._dm.base_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
