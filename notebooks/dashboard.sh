#!/bin/bash
# Worker dashboard — run from the notebooks/ directory
# Usage: ./dashboard.sh          (full dashboard with on-chain data)
#        ./dashboard.sh --fast   (local-only, skip on-chain sync)

cd "$(dirname "$0")"
source ../.venv/bin/activate

if [[ "$1" == "--fast" ]]; then
    python -m allora_forge_builder_kit.workerctl dashboard --no-monitor
else
    python -m allora_forge_builder_kit.workerctl dashboard
fi
