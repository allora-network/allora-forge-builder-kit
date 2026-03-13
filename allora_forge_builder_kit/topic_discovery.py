#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Allora Topic Discovery
======================

Query the list of available Allora Network topics and their configuration
using the allora_sdk API client.  This enables agents to discover all
active topics, understand their parameters (price vs. log-return,
prediction horizon, etc.), and submit appropriately formatted predictions.

Requires:
    pip install allora_sdk
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TopicInfo:
    """Parsed metadata for a single Allora topic."""

    topic_id: int
    metadata: str
    epoch_length: int
    ground_truth_lag: int
    loss_method: str
    is_active: bool
    raw: Dict[str, Any]

    @property
    def description(self) -> str:
        return self.metadata

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return (
            f"TopicInfo(id={self.topic_id}, status={status}, "
            f"epoch_length={self.epoch_length}, "
            f"loss='{self.loss_method}', "
            f"metadata='{self.metadata[:80]}...')"
        )


class AlloraTopicDiscovery:
    """
    Discover and inspect Allora Network topics.

    Uses the allora_sdk ``AlloraAPIClient`` under the hood.

    Example::

        discovery = AlloraTopicDiscovery(api_key="UP-...")
        topics = discovery.get_all_topics()
        for t in topics:
            print(t.topic_id, t.metadata)

        topic = discovery.get_topic(69)
        print(topic.epoch_length)
    """

    def __init__(self, api_key: Optional[str] = None, network: str = "testnet"):
        try:
            from allora_sdk.api_client import AlloraAPIClient, ChainID
        except ImportError:
            raise ImportError(
                "allora_sdk is required for topic discovery.  "
                "Install it with:  pip install allora_sdk"
            )
        chain_id = ChainID.MAINNET if network.lower() == "mainnet" else ChainID.TESTNET
        self._client = AlloraAPIClient(chain_id=chain_id, api_key=api_key)
        self._topics_cache: Optional[List[TopicInfo]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_all_topics(self, refresh: bool = False) -> List[TopicInfo]:
        """Return every active topic on the network (cached after first call)."""
        if self._topics_cache is None or refresh:
            self._topics_cache = _run(self._async_get_all_topics())
        return self._topics_cache

    def get_topic(self, topic_id: int) -> Optional[TopicInfo]:
        """Fetch a single topic by ID, or ``None`` if not found."""
        for t in self.get_all_topics():
            if t.topic_id == topic_id:
                return t
        return None

    def get_inference(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Fetch the latest network inference for a topic."""
        return _run(self._async_get_inference(topic_id))

    def list_price_topics(self) -> List[TopicInfo]:
        """Filter topics that predict an absolute price."""
        return [
            t
            for t in self.get_all_topics()
            if "price" in t.metadata.lower() and "log" not in t.metadata.lower()
        ]

    def list_log_return_topics(self) -> List[TopicInfo]:
        """Filter topics that predict log returns."""
        return [
            t
            for t in self.get_all_topics()
            if "log" in t.metadata.lower() and "return" in t.metadata.lower()
        ]

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------
    async def _async_get_all_topics(self) -> List[TopicInfo]:
        raw_topics = await self._client.get_all_topics()
        return [self._parse_topic(t) for t in raw_topics]

    async def _async_get_inference(self, topic_id: int) -> Optional[Dict[str, Any]]:
        try:
            result = await self._client.get_inference_by_topic_id(topic_id)
            return {
                "topic_id": topic_id,
                "network_inference": getattr(
                    getattr(result, "inference_data", None),
                    "network_inference_normalized",
                    None,
                ),
                "raw": result,
            }
        except Exception as e:
            print(f"[topic discovery] inference error for topic {topic_id}: {e}")
            return None

    @staticmethod
    def _parse_topic(raw: Any) -> TopicInfo:
        """Extract a TopicInfo from the SDK topic object."""
        raw_dict = raw.__dict__ if hasattr(raw, "__dict__") else {}
        return TopicInfo(
            topic_id=getattr(raw, "topic_id", getattr(raw, "id", 0)),
            metadata=getattr(raw, "metadata", ""),
            epoch_length=getattr(raw, "epoch_length", 0),
            ground_truth_lag=getattr(raw, "ground_truth_lag", 0),
            loss_method=getattr(raw, "loss_method", ""),
            is_active=True,
            raw=raw_dict,
        )


# ------------------------------------------------------------------
# Utility: run async in sync context
# ------------------------------------------------------------------
def _run(coro):
    """Execute an async coroutine from synchronous code.

    Handles Jupyter/IPython environments where an event loop is already
    running by patching it with nest_asyncio (preferred) or falling back
    to a thread-based approach.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)

    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
