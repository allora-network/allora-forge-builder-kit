# Allora Data Exploration Skill

Explore and fetch market data from the Atlas data service and discover
Allora Network topics.

## When to use

- User wants to browse available datasets or topics.
- User needs to inspect historical OHLCV data.
- User wants to understand topic parameters before building a model.

## Atlas Data Exploration

```python
from allora_forge_builder_kit import AtlasDataManager

dm = AtlasDataManager(api_key="UP-...")

# List all Tiingo 1-minute datasets
datasets = dm.list_available_datasets(source="tiingo", frequency="1min")
for d in datasets:
    print(d["name"], d["metadata"].get("ticker"))

# Free-text search
results = dm.search_datasets("bitcoin")
for r in results:
    print(r["id"], r["name"], r["description"])

# Fetch recent data
df = dm.get_live_1min_data("btcusd", hours_back=4)
print(df.tail())
```

## Topic Discovery

```python
from allora_forge_builder_kit import AlloraTopicDiscovery

discovery = AlloraTopicDiscovery(api_key="UP-...")

# All active topics
for t in discovery.get_all_topics():
    print(f"Topic {t.topic_id}: {t.metadata[:60]}")

# Filter by type
price_topics = discovery.list_price_topics()
log_return_topics = discovery.list_log_return_topics()

# Latest inference for a topic
inf = discovery.get_inference(topic_id=69)
print(inf)
```

## Key points

- Atlas base URL: `https://forge-data.allora.run/api/`
- Auth: `X-API-Key` header (same key as legacy Allora service).
- Datasets follow naming pattern `tiingo_{ticker}_1min`.
- 76 Tiingo 1-minute datasets currently available.
- Data may lag wall-clock time; the manager handles this automatically.
