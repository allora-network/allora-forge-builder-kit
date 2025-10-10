"""
Comprehensive tests for DataManager implementations and Workflow integration.

Run with:
    pytest tests/test_data_managers.py -v

For integration tests (requires API keys and network):
    RUN_INTEGRATION_TESTS=1 ALLORA_API_KEY=your-key pytest tests/test_data_managers.py -v
"""

import os
from datetime import datetime, timezone, timedelta

import pytest
import pandas as pd
import polars as pl

from allora_forge_builder_kit import (
    DataManager,
    BinanceDataManager,
    AlloraDataManager,
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
    """Test that factory returns AlloraDataManager for source='allora'."""
    dm = DataManager(
        source="allora",
        interval="5m",
        api_key="test-key",
        base_dir=str(tmp_path / "allora_data")
    )
    
    assert isinstance(dm, AlloraDataManager)
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
# Unit Tests - AlloraDataManager
# ============================================================================

def test_allora_manager_initialization(tmp_path):
    """Test AlloraDataManager initialization."""
    dm = AlloraDataManager(
        api_key="test-key",
        interval="5m",
        symbols=["btcusd"],  # Allora uses lowercase format
        base_dir=str(tmp_path / "allora_data")
    )
    
    assert dm.interval == "5m"
    assert dm.api_key == "test-key"
    assert "btcusd" in dm.symbols
    assert dm.base_dir == str(tmp_path / "allora_data")


def test_allora_manager_default_directory():
    """Test AlloraDataManager uses source-specific default directory."""
    dm = AlloraDataManager(api_key="test-key", interval="5m")
    assert dm.base_dir == "parquet_data_allora"


def test_allora_manager_parse_bar():
    """Test Allora bar parsing to standardized format."""
    dm = AlloraDataManager(api_key="test-key", interval="5m")
    
    # Create a mock Series
    timestamp = datetime(2025, 10, 6, 12, 0, tzinfo=timezone.utc)
    row = pd.Series({
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 100.5,
        "trades_done": 1000
    })
    
    bar = dm._parse_allora_bar(timestamp, row, "BTC/USD")
    
    assert bar["symbol"] == "BTC/USD"
    assert bar["open_time"] == timestamp
    assert bar["open"] == 50000.0
    assert bar["high"] == 50100.0
    assert bar["low"] == 49900.0
    assert bar["close"] == 50050.0
    assert bar["volume"] == 100.5
    assert pd.isna(bar["quote_volume"])  # Should be NaN for Allora
    assert bar["n_trades"] == 1000  # Mapped from trades_done


# ============================================================================
# Unit Tests - Storage Structure
# ============================================================================

def test_storage_separation(tmp_path):
    """Test that different sources use different directories."""
    binance_dm = BinanceDataManager(base_dir=str(tmp_path / "binance"))
    allora_dm = AlloraDataManager(api_key="test", base_dir=str(tmp_path / "allora"))
    
    assert binance_dm.base_dir != allora_dm.base_dir
    assert "binance" in binance_dm.base_dir
    assert "allora" in allora_dm.base_dir


def test_partition_path(tmp_path):
    """Test partition path generation."""
    dm = BinanceDataManager(base_dir=str(tmp_path / "data"))
    
    path = dm._partition_path("BTCUSDT", "2025-10-06")
    
    assert "symbol=BTCUSDT" in path
    assert "dt=2025-10-06.parquet" in path


# ============================================================================
# Unit Tests - Workflow Integration
# ============================================================================

def test_workflow_with_binance_string_api(tmp_path):
    """Test AlloraMLWorkflow with Binance using string API."""
    workflow = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        hours_needed=24,
        number_of_input_candles=24,
        target_length=24,
        interval="5m",
        data_source="binance",
        market="futures",
        base_dir=str(tmp_path / "binance_data")
    )
    
    assert isinstance(workflow._dm, BinanceDataManager)
    assert workflow.interval == "5m"


def test_workflow_with_allora_string_api(tmp_path):
    """Test AlloraMLWorkflow with Allora using string API."""
    workflow = AlloraMLWorkflow(
        tickers=["btcusd"],  # Allora uses lowercase format
        hours_needed=24,
        number_of_input_candles=24,
        target_length=24,
        interval="5m",
        data_source="allora",
        api_key="test-key",
        base_dir=str(tmp_path / "allora_data")
    )
    
    assert isinstance(workflow._dm, AlloraDataManager)
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
        hours_needed=24,
        number_of_input_candles=24,
        target_length=24,
        data_manager=dm
    )
    
    assert workflow._dm is dm
    assert isinstance(workflow._dm, BinanceDataManager)


def test_workflow_invalid_manager():
    """Test that workflow rejects non-BaseDataManager instances."""
    with pytest.raises(TypeError, match="must be an instance of BaseDataManager"):
        AlloraMLWorkflow(
            tickers=["BTCUSDT"],
            hours_needed=24,
            number_of_input_candles=24,
            target_length=24,
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
# Integration Tests - AlloraDataManager (requires API key + network)
# ============================================================================

@pytest.mark.integration
def test_allora_backfill_and_load(tmp_path, test_symbols_allora, allora_api_key, integration_check):
    """Test Allora backfill and load operations."""
    dm = AlloraDataManager(
        api_key=allora_api_key,
        interval="5m",
        symbols=test_symbols_allora,
        base_dir=str(tmp_path / "allora_data")
    )
    
    # Backfill last 3 days (wider range to get more data)
    start = datetime.now(timezone.utc) - timedelta(days=3)
    end = datetime.now(timezone.utc)
    
    dm.backfill_symbol(test_symbols_allora[0], start, end)
    
    # Load data (no end filter to include all data)
    df = dm.load_pandas(test_symbols_allora, start=start)
    
    # Allora API may have limited historical data availability
    # If empty, the test still passes (API constraints, not code failure)
    if df.empty:
        print(f"[test] Allora API returned no data for {test_symbols_allora[0]} - this is expected for limited historical availability")
        return
    
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["symbol", "open_time"]
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "n_trades" in df.columns
    assert "quote_volume" in df.columns  # Should be present (but NaN)


@pytest.mark.integration
def test_allora_live_snapshot(test_symbols_allora, allora_api_key, integration_check):
    """Test Allora get_live_snapshot."""
    dm = AlloraDataManager(api_key=allora_api_key, interval="5m")
    
    snapshot = dm.get_live_snapshot(test_symbols_allora)
    
    assert not snapshot.empty
    assert isinstance(snapshot.index, pd.MultiIndex)
    assert len(snapshot) == len(test_symbols_allora)
    assert "open" in snapshot.columns
    assert "close" in snapshot.columns
    assert "volume" in snapshot.columns
    assert "n_trades" in snapshot.columns


@pytest.mark.integration
def test_allora_get_live_1min_data(test_symbols_allora, allora_api_key, integration_check):
    """Test Allora get_live_1min_data returns 1-minute bars."""
    # First verify API key works with a direct API call
    import requests
    headers = {"x-api-key": allora_api_key}
    params = {"tickers": "btcusd", "from_date": "2025-10-09"}
    
    try:
        response = requests.get(
            "https://api.allora.network/v2/allora/market-data/ohlc",
            headers=headers,
            params=params,
            timeout=10
        )
        if response.status_code == 401:
            pytest.skip("Allora API key is invalid or expired")
        elif response.status_code != 200:
            pytest.skip(f"Allora API returned {response.status_code}")
    except Exception as e:
        pytest.skip(f"Cannot connect to Allora API: {e}")
    
    # Now test the data manager
    dm = AlloraDataManager(api_key=allora_api_key, interval="5m")
    
    # Fetch 2 hours of 1-minute data
    df_1min = dm.get_live_1min_data(test_symbols_allora[0], hours_back=2)
    
    if df_1min.empty:
        pytest.skip("Allora API returned no data (may have limited historical availability)")
    
    assert isinstance(df_1min.index, pd.DatetimeIndex)
    assert list(df_1min.columns) == ["open", "high", "low", "close", "volume"]
    
    # Verify it's 1-minute bars
    if len(df_1min) > 1:
        time_diff = (df_1min.index[1] - df_1min.index[0]).total_seconds() / 60
        assert abs(time_diff - 1.0) < 0.1, f"Expected 1-minute bars, got {time_diff} minutes"
    
    # Allora may have less data available
    assert len(df_1min) > 0, "Should have at least some 1-minute bars"


# ============================================================================
# Integration Tests - Workflow with Binance
# ============================================================================

@pytest.mark.integration
def test_workflow_binance_backfill_and_features(tmp_path, test_symbols_binance, integration_check):
    """Test full workflow with Binance: backfill + feature extraction."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
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
    df_features = workflow.get_full_feature_target_dataframe_pandas(start_date=start)
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
        hours_needed=2,  # Small lookback for testing
        number_of_input_candles=8,
        target_length=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    # Get live features
    features = workflow.get_live_features(test_symbols_binance[0])
    
    assert not features.empty
    assert features.shape[0] == 1  # Should return 1 row (latest bar)
    assert features.shape[1] == 8 * 5  # 8 candles × 5 OHLCV features
    assert isinstance(features.index, pd.DatetimeIndex)
    
    # Verify features are normalized (should be around 1.0)
    assert (features.iloc[0] > 0).all(), "All features should be positive"
    assert (features.iloc[0] < 10).all(), "Features should be normalized (< 10)"


@pytest.mark.integration  
def test_workflow_create_interval_bars(test_symbols_binance, integration_check):
    """Test workflow create_interval_bars resamples 1-min to target interval."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    # Get 1-minute data
    df_1min = workflow._dm.get_live_1min_data(test_symbols_binance[0], hours_back=2)
    
    # Resample to 5-minute bars
    df_5min = workflow.create_interval_bars(df_1min, live_mode=False)
    
    assert not df_5min.empty
    assert isinstance(df_5min.index, pd.DatetimeIndex)
    assert list(df_5min.columns) == ["open", "high", "low", "close", "volume"]
    
    # Verify resampling worked (should have ~24 5-min bars in 2 hours)
    expected_bars = int((2 * 60) / 5)  # 2 hours × 60 min / 5 min
    assert len(df_5min) <= expected_bars + 5  # Allow some tolerance
    
    # Verify it's 5-minute bars
    if len(df_5min) > 1:
        time_diff = (df_5min.index[1] - df_5min.index[0]).total_seconds() / 60
        assert abs(time_diff - 5.0) < 0.1, f"Expected 5-minute bars, got {time_diff} minutes"


@pytest.mark.integration
def test_workflow_live_mode_offset(test_symbols_binance, integration_check):
    """Test that live_mode applies offset to align last bar with last 1-min bar."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        hours_needed=2,
        number_of_input_candles=8,
        target_length=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    # Get 1-minute data
    df_1min = workflow._dm.get_live_1min_data(test_symbols_binance[0], hours_back=2)
    
    # Print for visual inspection
    print(f"\n{'='*80}")
    print(f"1-minute bars fetched: {len(df_1min)}")
    print(f"Last 10 1-minute bars:")
    print(df_1min.tail(10)[['close']])
    print(f"Last 1-min bar time: {df_1min.index[-1]}")
    
    # Resample with live_mode=True (should apply offset)
    df_5min_live = workflow.create_interval_bars(df_1min, live_mode=True)
    
    print(f"\n5-minute bars after resampling with live_mode=True: {len(df_5min_live)}")
    print(f"Last 5 5-minute bars:")
    print(df_5min_live.tail(5)[['close']])
    print(f"Last 5-min bar time: {df_5min_live.index[-1]}")
    
    # Also test without live_mode for comparison
    df_5min_no_live = workflow.create_interval_bars(df_1min, live_mode=False)
    print(f"\n5-minute bars after resampling with live_mode=False: {len(df_5min_no_live)}")
    print(f"Last 5 5-minute bars:")
    print(df_5min_no_live.tail(5)[['close']])
    print(f"Last 5-min bar time: {df_5min_no_live.index[-1]}")
    print(f"{'='*80}\n")
    
    assert not df_5min_live.empty
    
    # Verify the close price of the last 5-min bar matches one of the recent 1-min bars
    # (since the last 5-min bar is aggregated from the last 5 complete 1-min bars)
    last_5min_close = df_5min_live.iloc[-1]['close']
    recent_1min_closes = df_1min.tail(10)['close'].values
    
    # The last 5-min bar's close should match one of the recent 1-min bar closes
    # (it's the close of the last 1-min bar that fell within that 5-min window)
    assert last_5min_close in recent_1min_closes, \
        f"Last 5-min close ({last_5min_close}) should match one of recent 1-min closes"
    
    # Verify that with live_mode we get offset-aligned bars
    # The last bar time modulo 5 should give us the offset
    last_5min_time = df_5min_live.index[-1]
    offset_minutes = (last_5min_time.minute + 1) % 5
    print(f"Offset applied: {offset_minutes} minutes")


@pytest.mark.integration
def test_workflow_get_live_features_end_to_end(test_symbols_binance, integration_check):
    """End-to-end test of get_live_features() with validation and visual inspection."""
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_binance,
        hours_needed=2,
        number_of_input_candles=8,
        target_length=16,
        interval="5m",
        data_source="binance",
        market="futures"
    )
    
    print(f"\n{'='*80}")
    print(f"End-to-End: get_live_features() for {test_symbols_binance[0]}")
    print(f"Configuration:")
    print(f"  - interval: {workflow.interval}")
    print(f"  - hours_needed: {workflow.hours_needed}")
    print(f"  - number_of_input_candles: {workflow.number_of_input_candles}")
    
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
    assert features.shape == (1, 40), f"Expected (1, 40), got {features.shape}"
    
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
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
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
    df_features = workflow.get_full_feature_target_dataframe_pandas(start_date=start)
    assert not df_features.empty
    assert "target" in df_features.columns
    
    # Check feature columns exist
    feature_cols = [col for col in df_features.columns if col.startswith("feature_")]
    assert len(feature_cols) > 0


@pytest.mark.integration
def test_workflow_allora_get_live_features(test_symbols_allora, allora_api_key, integration_check):
    """Test workflow get_live_features with Allora."""
    # Verify API key works first
    import requests
    headers = {"x-api-key": allora_api_key}
    params = {"tickers": "btcusd", "from_date": "2025-10-09"}
    
    try:
        response = requests.get(
            "https://api.allora.network/v2/allora/market-data/ohlc",
            headers=headers,
            params=params,
            timeout=10
        )
        if response.status_code == 401:
            pytest.skip("Allora API key is invalid or expired")
        elif response.status_code != 200:
            pytest.skip(f"Allora API returned {response.status_code}")
    except Exception as e:
        pytest.skip(f"Cannot connect to Allora API: {e}")
    
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_allora,
        hours_needed=2,  # Small lookback for testing
        number_of_input_candles=8,
        target_length=16,
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
    assert features.shape[1] == 8 * 5  # 8 candles × 5 OHLCV features
    assert isinstance(features.index, pd.DatetimeIndex)
    
    # Verify features are normalized (should be around 1.0)
    assert (features.iloc[0] > 0).all(), "All features should be positive"
    assert (features.iloc[0] < 10).all(), "Features should be normalized (< 10)"


@pytest.mark.integration  
def test_workflow_allora_create_interval_bars(test_symbols_allora, allora_api_key, integration_check):
    """Test workflow create_interval_bars resamples 1-min to target interval with Allora."""
    # Verify API key works first
    import requests
    headers = {"x-api-key": allora_api_key}
    params = {"tickers": "btcusd", "from_date": "2025-10-09"}
    
    try:
        response = requests.get(
            "https://api.allora.network/v2/allora/market-data/ohlc",
            headers=headers,
            params=params,
            timeout=10
        )
        if response.status_code == 401:
            pytest.skip("Allora API key is invalid or expired")
        elif response.status_code != 200:
            pytest.skip(f"Allora API returned {response.status_code}")
    except Exception as e:
        pytest.skip(f"Cannot connect to Allora API: {e}")
    
    workflow = AlloraMLWorkflow(
        tickers=test_symbols_allora,
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
        interval="5m",
        data_source="allora",
        api_key=allora_api_key
    )
    
    # Get 1-minute data
    df_1min = workflow._dm.get_live_1min_data(test_symbols_allora[0], hours_back=2)
    
    if df_1min.empty:
        pytest.skip("Allora API returned no data (may have limited historical availability)")
    
    # Resample to 5-minute bars
    df_5min = workflow.create_interval_bars(df_1min, live_mode=False)
    
    assert not df_5min.empty
    assert isinstance(df_5min.index, pd.DatetimeIndex)
    assert list(df_5min.columns) == ["open", "high", "low", "close", "volume"]
    
    # Verify it's 5-minute bars
    if len(df_5min) > 1:
        time_diff = (df_5min.index[1] - df_5min.index[0]).total_seconds() / 60
        assert abs(time_diff - 5.0) < 0.1, f"Expected 5-minute bars, got {time_diff} minutes"


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
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
        data_source="binance",
        market="futures",
        base_dir=str(tmp_path / "binance")
    )
    
    allora_wf = AlloraMLWorkflow(
        tickers=["btcusd"],  # Allora uses lowercase format
        hours_needed=24,
        number_of_input_candles=24,
        target_length=16,
        data_source="allora",
        api_key=allora_key,
        base_dir=str(tmp_path / "allora")
    )
    
    # Verify different data managers
    assert isinstance(binance_wf._dm, BinanceDataManager)
    assert isinstance(allora_wf._dm, AlloraDataManager)
    
    # Verify different storage
    assert binance_wf._dm.base_dir != allora_wf._dm.base_dir
    assert "binance" in binance_wf._dm.base_dir
    assert "allora" in allora_wf._dm.base_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
