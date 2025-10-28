# V2.0 Release Critical Updates

## ✅ COMPLETED: Fixed Time Series Split in example_topic_69_bitcoin_prediction.py

### What Was Wrong
- Naive percentage-based split (70/15/15) without embargo gap
- Split on row indices instead of unique dates
- **CRITICAL BUG**: Model could "see the future" through data leakage

### What Was Fixed
- ✅ Proper `TimeSeriesSplit` with embargo gap = target_length
- ✅ Split on unique dates (not rows) to handle multi-asset scenarios
- ✅ 2-split strategy: Split 1 → train/val, Split 2 → test
- ✅ Added sklearn import

### The Fix
```python
from sklearn.model_selection import TimeSeriesSplit

# Calculate embargo gap
embargo_gap = int(TARGET_LENGTH * 60 / minute_cadence)  # 24 intervals for 24h

# Split on unique dates
unique_dates = sorted(df_all['open_time'].unique())
n_dates = len(unique_dates)

# Enforce 60/20/20 split
min_train_size = int(0.6 * n_dates)

# TimeSeriesSplit with gap and min_train_size
tscv = TimeSeriesSplit(
    n_splits=2, 
    gap=embargo_gap,
    min_train_size=min_train_size  # Force 60% training
)
splits = list(tscv.split(unique_dates))

# Split 1: train (60%) → val (20%), Split 2: train (80%) → test (20%)
train_idx_1, val_idx = splits[0]
_, test_idx = splits[1]
```

---

## ⚠️ TODO: Update Topic 69 Notebook

The notebook `Topic 69 - Bitcoin Price Prediction.ipynb` needs the same fix applied manually.

**Cell to Update**: The cell with `# 70% train, 15% validation, 15% test`

**Replacement Code**: Copy from `example_topic_69_bitcoin_prediction.py` lines 109-170

---

## 🎬 Animation Templates for V2 Features

### Animation 1: Data Manager Architecture
**Title**: "Multi-Source Data Management"
**Duration**: ~30 seconds

**Scenes**:
1. Logo + Title fade in
2. Code: Initialize BinanceDataManager
   ```python
   from allora_forge_builder_kit import DataManager
   
   # Option 1: Binance (no API key)
   dm_binance = DataManager(
       source="binance",
       market="futures",
       interval="5m"
   )
   ```
3. Message: "Switch sources instantly..."
4. Code: Initialize AlloraDataManager
   ```python
   # Option 2: Allora Network (with API key)
   dm_allora = DataManager(
       source="allora",
       api_key=ALLORA_API_KEY,
       interval="5m"
   )
   ```
5. Message: "Same API. Different data."
6. Code: Show unified interface
   ```python
   # Unified interface - works with both!
   dm.backfill(tickers=["btcusd"], start=start_date)
   data = dm.load_raw(start=start, end=end)
   ```
7. Logo fade in


### Animation 2: Proper Time Series Splitting
**Title**: "Prevent Data Leakage"
**Duration**: ~35 seconds

**Scenes**:
1. Logo + Title fade in
2. Message: "Traditional splits leak data ❌"
3. Code: Show WRONG way (highlighted as bad)
   ```python
   # ❌ WRONG: Naive split leaks future data
   train_end = int(len(df) * 0.70)
   X_train = df.iloc[:train_end]
   X_test = df.iloc[train_end:]
   # Model can "see" 24h into the future!
   ```
4. Message: "Add embargo gap ✓"
5. Code: Show RIGHT way
   ```python
   # ✓ CORRECT: Embargo prevents leakage
   embargo_gap = int(TARGET_LENGTH * 60 / minute_cadence)
   
   tscv = TimeSeriesSplit(n_splits=2, gap=embargo_gap)
   unique_dates = sorted(df['open_time'].unique())
   splits = list(tscv.split(unique_dates))
   
   # Now model CAN'T see the future
   ```
6. Visual: Show timeline with gap
7. Logo fade in


### Animation 3: Live Feature Extraction
**Title**: "Fresh Data, Every Time"
**Duration**: ~25 seconds

**Scenes**:
1. Logo + Title fade in
2. Message: "One line gets live features"
3. Code: Show the magic
   ```python
   # Automatically:
   # • Fetches fresh 1-min data
   # • Drops incomplete bars
   # • Resamples to your interval
   # • Extracts normalized features
   
   features = workflow.get_live_features("btcusd")
   # Shape: (1, 120) - ready for model.predict()
   ```
4. Message: "Behind the scenes..."
5. Show process indicators (like current animation)
   - ⏳ Fetching 1-min data...
   - ⏳ Resampling to 1h...
   - ⏳ Extracting features...
   - ✓ Ready in 0.3s
6. Logo fade in


### Animation 4: Fast Allora Loading
**Title**: "Monthly Buckets = Speed"
**Duration**: ~20 seconds

**Scenes**:
1. Logo + Title fade in
2. Split screen comparison:
   **Left**: Binance (slow)
   ```python
   # Binance: 1000-bar limit
   # 6 months = 180 API calls
   # ~3 minutes
   ```
   **Right**: Allora (fast)
   ```python
   # Allora: Monthly buckets
   # 6 months = 6 API calls
   # ~10 seconds
   ```
3. Message: "18x faster data loading"
4. Logo fade in

---

---

## ✅ NEW FEATURE: Official Performance Metrics (ENGN-4244)

### What Was Added
Comprehensive performance evaluation system aligned with Research team's framework (RES-1087).

### The Implementation
New `PerformanceEvaluator` class in `allora_forge_builder_kit/evaluation.py`:

```python
from allora_forge_builder_kit import PerformanceEvaluator

evaluator = PerformanceEvaluator()
report = evaluator.evaluate(
    y_true=actual_log_returns,
    y_pred=predicted_log_returns,
    epoch_length_minutes=60
)

evaluator.print_report(report)
```

### 8 Primary Metrics (with pass/fail thresholds)
1. **Directional Accuracy** (≥ 0.55) - % correct sign predictions
2. **DA Confidence Interval Lower** (≥ 0.52) - Statistical confidence
3. **DA p-value** (< 0.05) - Statistical significance
4. **Pearson Correlation** (≥ 0.05) - Linear relationship
5. **Pearson p-value** (< 0.05) - Correlation significance  
6. **WRMSE Improvement** (≥ 0.10) - 10% improvement vs baseline
7. **ZPTAE Improvement** (≥ 0.20) - 20% improvement (robust loss)
8. **Log Aspect Ratio** (|value| < 0.5) - Variance matching

### Performance Grading
- **8/8 passed** → Grade: A+ (Score: 1.0)
- **7/8 passed** → Grade: A (Score: 0.875)
- **6/8 passed** → Grade: B+ (Score: 0.75)
- **< 4/8 passed** → Grade: F

### Additional Metrics
- Regression: MAE, MSE, RMSE, R², MAPE
- Classification: Precision, Recall, F1, Specificity
- Trading: Naive Annualized Return
- Correlation: Spearman rank correlation

### Integration
✅ Added to `example_topic_69_bitcoin_prediction.py` (Step 6.5)
✅ Exported from main package (`from allora_forge_builder_kit import PerformanceEvaluator`)
✅ Fully documented with comprehensive report printing

---

## 📋 Release Checklist

### Code Updates
- [x] Fix `example_topic_69_bitcoin_prediction.py` with proper TimeSeriesSplit (60/20/20)
- [x] Add PerformanceEvaluator for official metrics (ENGN-4244)
- [x] Integrate metrics into example script
- [ ] Fix `Topic 69 - Bitcoin Price Prediction.ipynb` (same updates)
- [x] sklearn already in requirements.txt
- [ ] Add scipy to requirements.txt (for stats)
- [ ] Update CHANGELOG.md with all V2 features
- [ ] Update README.md to mention proper splitting and metrics

### Documentation
- [ ] Add section to AGENT_GUIDE.md about time series splitting
- [ ] Create example in README showing TimeSeriesSplit
- [ ] Update Topic 69 documentation

### Animations
- [ ] Create Animation 1: Data Manager (use workflow_config_full.html as template)
- [ ] Create Animation 2: Time Series Split (new template)
- [ ] Create Animation 3: Live Features (already have base in live_features.html)
- [ ] Create Animation 4: Fast Loading (simple comparison)

### Testing
- [ ] Run updated example_topic_69_bitcoin_prediction.py
- [ ] Verify embargo gap is correct
- [ ] Check that dates don't overlap
- [ ] Validate model performance doesn't change significantly

---

## 🚨 Critical Note

**The time series split fix is CRITICAL for production models.** Without the embargo gap, models will have unrealistically good validation metrics but fail in live trading because they've been trained on future data.

### Impact
- Old split: Model "sees" 24 hours into future → optimistic metrics
- New split: Model respects causality → realistic metrics

This may cause validation scores to drop, but that's **EXPECTED and CORRECT**.


