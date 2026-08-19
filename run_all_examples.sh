#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
NOTEBOOKS="$REPO_DIR/notebooks/testnet"

run_script() {
    local dir="$1"
    local script="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  RUNNING: $dir/$script"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    cd "$NOTEBOOKS/$dir"
    "$PYTHON" "$script"
    echo ""
    echo "  ✓ Done: $dir/$script"
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

# ── topic_57 ─────────────────────────────────────────────────────────────────
run_script topic_57_sol_8h_logreturn example.py

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
echo "  ALL SCRIPTS COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
