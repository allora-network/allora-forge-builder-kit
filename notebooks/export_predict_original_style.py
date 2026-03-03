#!/usr/bin/env python3
"""DEPRECATED: intentionally disabled.

This exporter used a raw-data inference path (`load_raw`) that is not safe for
live worker deployment and can cause stale/failed submissions.

Use `export_predict_self_contained.py` instead.
"""

raise SystemExit(
    "❌ export_predict_original_style.py is disabled. "
    "Use notebooks/export_predict_self_contained.py for deployable artifacts."
)
