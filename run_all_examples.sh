#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
NOTEBOOKS="$REPO_DIR/notebooks/testnet"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    echo "Activate the project virtualenv or set PYTHON=/path/to/python." >&2
    exit 2
fi

if [[ -z "${ALLORA_API_KEY//[[:space:]]/}" ]] && ! grep -q '[^[:space:]]' "$REPO_DIR/.allora_api_key" 2>/dev/null; then
    echo "Allora API key not found." >&2
    echo "Set ALLORA_API_KEY or create $REPO_DIR/.allora_api_key before running the example suite." >&2
    exit 2
fi

FAILED=()
PASSED=()

run_script() {
    local dir="$1"
    local script="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  RUNNING: $dir/$script"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    cd "$NOTEBOOKS/$dir"
    if "$PYTHON" "$script" </dev/null; then
        echo ""
        echo "  ✓ Done: $dir/$script"
        PASSED+=("$dir/$script")
    else
        echo ""
        echo "  ✗ FAILED (exit $?): $dir/$script"
        FAILED+=("$dir/$script")
    fi
}

# ── topic_38 ─────────────────────────────────────────────────────────────────
run_script topic_38_sol_8h_price example.py
run_script topic_38_sol_8h_price model_czar.py
run_script topic_38_sol_8h_price model_v3_methodology.py

# ── topic_41 ─────────────────────────────────────────────────────────────────
run_script topic_41_eth_8h_price example.py
run_script topic_41_eth_8h_price model_czar.py

# ── topic_42 ─────────────────────────────────────────────────────────────────
run_script topic_42_btc_8h_price example.py
run_script topic_42_btc_8h_price model_v2_directional.py
run_script topic_42_btc_8h_price model_v3_czar.py

# ── topic_58 ─────────────────────────────────────────────────────────────────
run_script topic_58_sol_8h_logreturn example.py

# ── topic_61 ─────────────────────────────────────────────────────────────────
run_script topic_61_btc_24h_logreturn example.py

# ── topic_62 ─────────────────────────────────────────────────────────────────
run_script topic_62_sol_24h_logreturn example.py

# ── topic_63 ─────────────────────────────────────────────────────────────────
run_script topic_63_eth_24h_logreturn example.py

# ── topic_71 ─────────────────────────────────────────────────────────────────
run_script topic_71_near_8h_logreturn example.py

# ── topic_79 ─────────────────────────────────────────────────────────────────
run_script topic_79_btc_vol model_grid_retrain.py

# ── topic_80 ─────────────────────────────────────────────────────────────────
run_script topic_80_eth_vol model_grid_retrain.py

# ── topic_81 ─────────────────────────────────────────────────────────────────
run_script topic_81_xrp_vol model_grid_retrain.py

# ── topic_82 ─────────────────────────────────────────────────────────────────
run_script topic_82_sol_vol model_grid_retrain.py

# ── topic_83 ─────────────────────────────────────────────────────────────────
run_script topic_83_btc_8h_logreturn example.py

# ── topic_84 ─────────────────────────────────────────────────────────────────
run_script topic_84_eth_8h_logreturn example.py

# ── topic_85 ─────────────────────────────────────────────────────────────────
run_script topic_85_eth_4h_vol model_grid_retrain.py
run_script topic_85_eth_4h_vol model_importance_groups.py

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  SUMMARY: ${#PASSED[@]} passed, ${#FAILED[@]} failed"
echo "════════════════════════════════════════════════════════════════════════"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Failed scripts:"
    for s in "${FAILED[@]}"; do
        echo "    ✗ $s"
    done
    echo ""
    exit 1
fi
