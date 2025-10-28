# Changelog

All notable changes to the Allora Forge Builder Kit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-10-15

### 🎉 Major Release - Complete Architecture Overhaul

### Added
- **Modular Data Management Architecture**
  - `BaseDataManager` abstract class for extensibility
  - `BinanceDataManager` for Binance Spot & Futures data
  - `AlloraDataManager` for Allora Network data
  - `DataManager()` factory function for easy instantiation
  - Storage isolation: separate `parquet_data_binance/` and `parquet_data_allora/` directories

- **Live Feature Extraction**
  - `get_live_features()` method for real-time inference
  - Automatic 1-minute data fetching from APIs
  - Smart resampling with offset alignment
  - Data-source agnostic implementation (works with any data manager)
  - In-memory caching for performance

- **Comprehensive Testing Suite**
  - 17 unit tests (data manager functionality)
  - 6 integration tests (live API validation)
  - Test framework with `RUN_INTEGRATION_TESTS` flag
  - Visual inspection tests for offset logic validation
  - End-to-end workflow validation

- **Documentation**
  - `AGENT_GUIDE.md`: Comprehensive guide for AI agents
  - `ARCHITECTURE.md`: Deep dive into system design
  - `tests/README.md`: Test documentation with examples
  - Complete README overhaul with Topic 69 example
  - Multiple example notebooks and scripts

- **Performance Optimizations**
  - Numba JIT compilation for feature extraction
  - Polars integration for fast data operations
  - Efficient Parquet storage with daily partitioning
  - Smart gap detection and backfill
  - In-memory caching with configurable cache length

### Changed
- **Breaking**: Renamed package from `allora-ml-workflow` to `allora-forge-builder-kit`
- **Breaking**: New workflow initialization signature with `data_source` parameter
- **Breaking**: Data storage moved to source-specific directories
- Workflow now uses factory pattern for data manager creation
- Feature extraction significantly optimized with Numba
- Improved error handling and logging throughout

### Technical Details
- Live feature extraction bypasses local storage for always-fresh data
- Offset resampling ensures bars align perfectly with last completed minute
- Monthly bucket support for Allora (much faster than Binance)
- WebSocket streaming support for Binance (live bar completion callbacks)
- Automatic deduplication in Parquet storage

### Migration Guide (v0.1 → v2.0)

**Old (v0.1):**
```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    interval="5m"
)
```

**New (v2.0):**
```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    interval="5m",
    data_source="binance",  # NEW: specify source
    market="futures"         # Optional: spot or futures
)
```

### Notes
- Old notebook outputs may show `0.1.0` - this is expected (cached execution results)
- Data from v0.1 in `parquet_data/` can be migrated to `parquet_data_binance/`
- No changes required to model training or prediction code

---

## [0.1.0] - 2025-04-XX (Approximate)

### Initial Release
- Basic workflow for ML model training on cryptocurrency data
- Simple data fetching from Allora API
- Feature extraction with OHLCV normalization
- Target calculation for prediction horizons
- Basic Parquet storage
- Example notebooks for model training

---

[2.0.0]: https://github.com/allora-network/allora-forge-builder-kit/compare/v0.1.0...v2.0.0
[0.1.0]: https://github.com/allora-network/allora-forge-builder-kit/releases/tag/v0.1.0



