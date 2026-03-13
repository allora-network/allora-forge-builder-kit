# Test Suite for Allora Forge Builder Kit

## Overview

Comprehensive test suite for the data manager architecture and ML workflow.

## Test Coverage

### Unit Tests (No network required)
- ✅ Factory pattern (DataManager routing)
- ✅ BinanceDataManager initialization
- ✅ AtlasDataManager initialization
- ✅ Data format parsing
- ✅ Storage separation
- ✅ Workflow integration

### Integration Tests (Requires network + API keys)
- ✅ Binance backfill and load
- ✅ Binance live snapshot
- ✅ Allora backfill and load (requires API key)
- ✅ Allora live snapshot (requires API key)
- ✅ Workflow with Binance (full pipeline)
- ✅ Workflow with Allora (full pipeline)
- ✅ Both sources coexist without collision

## Running Tests

### Run All Unit Tests (Fast, No Network)
```bash
pytest tests/test_data_managers.py -v -m "not integration"
```

### Run Unit Tests Only (Explicit)
```bash
pytest tests/test_data_managers.py -v -k "not test_binance_backfill and not test_allora and not test_workflow"
```

### Run All Tests Including Integration (Requires API keys)
```bash
# Set environment variables
export RUN_INTEGRATION_TESTS=1
export ALLORA_API_KEY="your-allora-api-key"

# Run all tests
pytest tests/test_data_managers.py -v
```

### Run Only Binance Integration Tests
```bash
export RUN_INTEGRATION_TESTS=1
pytest tests/test_data_managers.py -v -k "binance"
```

### Run Only Allora Integration Tests
```bash
export RUN_INTEGRATION_TESTS=1
export ALLORA_API_KEY="your-allora-api-key"
pytest tests/test_data_managers.py -v -k "allora"
```

### Run Specific Test
```bash
pytest tests/test_data_managers.py::test_factory_returns_binance_manager -v
```

## Test Organization

```
tests/
├── test_data_managers.py           # Data manager and workflow tests (30)
├── test_evaluation.py              # Evaluation metrics tests (18)
├── test_feature_integrity.py       # Feature normalization checks (3)
├── test_target_integrity.py        # Log-return target checks (3)
├── test_live_features_consistency.py  # Live vs historical (3, skipped)
└── README.md

Test Structure:
├── test_data_managers.py (30 tests)
│   ├── Unit Tests - Factory Pattern (4 tests)
│   ├── Unit Tests - BinanceDataManager (3 tests)
│   ├── Unit Tests - AtlasDataManager (2 tests)
│   ├── Unit Tests - Storage Structure (2 tests)
│   ├── Unit Tests - Workflow Integration (4 tests)
│   ├── Integration Tests - Binance (3 tests)
│   ├── Integration Tests - Atlas/Allora (3 tests)
│   ├── Integration Tests - Workflow Binance (3 tests)
│   ├── Integration Tests - Workflow Allora (3 tests)
│   └── Integration Tests - Compare Sources (1 test)
│   └── Integration Tests - End-to-End Atlas (1 test)
│   └── Integration Tests - End-to-End Binance (1 test)
├── test_evaluation.py (18 tests)
│   ├── Directional Accuracy (5 tests)
│   ├── Pearson Correlation (2 tests)
│   ├── WRMSE Improvement (2 tests)
│   ├── CZAR Improvement (3 tests)
│   ├── Threshold pass/fail (2 tests)
│   └── Full evaluate pipeline (4 tests)
├── test_feature_integrity.py (3 tests)
│   ├── 5-min bars feature integrity
│   ├── 1-hour bars feature integrity
│   └── Multi-asset feature integrity
├── test_target_integrity.py (3 tests)
│   ├── 5-min bars target integrity
│   ├── 1-hour bars target integrity
│   └── Multi-asset target integrity
├── test_live_features_consistency.py (3 tests, skipped)
│   └── Needs rewrite against current workflow API
└── README.md

Total: 57 tests (35 unit + 19 integration + 3 skipped)
```

## Requirements

### For Unit Tests
```
pytest
pandas
polars
```

### For Integration Tests
```
# Same as above, plus:
- Network connection
- Allora API key (for Allora tests)
```

## Test Markers

Tests are marked with:
- `@pytest.mark.integration` - Requires network and/or API keys

## Expected Results

### Unit Tests (Should Always Pass)
```
======================== test session starts =========================
tests/test_data_managers.py::test_factory_returns_binance_manager PASSED
tests/test_data_managers.py::test_factory_returns_allora_manager PASSED
tests/test_data_managers.py::test_factory_invalid_source PASSED
...
======================== 35 passed in 1.5s ==========================
```

### Integration Tests (May be slow)
```
======================== test session starts =========================
tests/test_data_managers.py::test_binance_backfill_and_load PASSED [10s]
tests/test_data_managers.py::test_allora_backfill_and_load PASSED [15s]
...
======================== 54 passed, 3 skipped in 45s ==========================
```

## Troubleshooting

### "ALLORA_API_KEY not set"
Set your API key:
```bash
export ALLORA_API_KEY="your-key-here"
```

### "Integration test. Set RUN_INTEGRATION_TESTS=1"
Enable integration tests:
```bash
export RUN_INTEGRATION_TESTS=1
```

### Network Timeout
Increase timeout in test if needed, or check internet connection.

### Test Data Location
Tests use `tmp_path` fixture, so data is stored in temporary directories and cleaned up automatically.

## CI/CD Integration

For CI pipelines:
```yaml
# Run unit tests only (fast)
- run: pytest tests/test_data_managers.py -v -m "not integration"

# Run integration tests (if API keys available)
- run: |
    export RUN_INTEGRATION_TESTS=1
    export ALLORA_API_KEY=${{ secrets.ALLORA_API_KEY }}
    pytest tests/test_data_managers.py -v
```

## What Each Test Validates

### Factory Pattern Tests
- Correct manager instantiation based on source string
- Error handling for invalid sources
- API key validation for Allora

### Manager Tests
- Proper initialization
- Default directory assignment
- Data parsing accuracy
- Standardized format compliance

### Storage Tests
- Directory separation between sources
- Partition path generation
- No collision between Binance and Allora data

### Workflow Tests
- Integration with both data sources
- String API functionality
- Explicit manager usage
- Feature extraction pipeline

### Integration Tests
- Real API calls work correctly
- Data is downloaded and stored
- Format is correct
- Full pipeline (backfill → load → features) works

## Notes

- Unit tests are fast and should always pass
- Integration tests require network and may be slower
- Allora tests require valid API key
- All tests use temporary directories for data
- Tests are independent (can run in any order)



