from __future__ import annotations

import argparse
import asyncio
import math
import os
import cloudpickle

from allora_sdk.worker import AlloraWorker
from allora_sdk.rpc_client.config import AlloraNetworkConfig, AlloraWalletConfig


def _load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("ALLORA_API_KEY")
    if env:
        return env
    for path in ("notebooks/.allora_api_key", ".allora_api_key"):
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
    raise RuntimeError("ALLORA_API_KEY not found")


_TESTNET_FAUCET_URL = "https://faucet.testnet.allora.run"


def _build_network(network: str, no_faucet: bool) -> AlloraNetworkConfig:
    cfg = AlloraNetworkConfig.mainnet() if network == "mainnet" else AlloraNetworkConfig.testnet()
    if no_faucet:
        cfg.faucet_url = None
    elif network != "mainnet":
        # SDK default points to .allora.network which 301-redirects to .allora.run;
        # requests silently converts the POST to a GET on redirect so the drip never fires.
        cfg.faucet_url = _TESTNET_FAUCET_URL
    return cfg


async def _run(
    topic_id: int,
    artifact_path: str,
    api_key: str,
    mnemonic_file: str | None = None,
    network: str = "testnet",
    no_faucet: bool = False,
    debug: bool = False,
    reject_zero: bool = False,
) -> None:
    with open(artifact_path, "rb") as f:
        raw_fn = cloudpickle.load(f)

    def run_fn(nonce: int):
        value = raw_fn(nonce)
        try:
            v = float(value)
        except Exception as e:
            raise RuntimeError(f"Invalid inference output type: {value!r}") from e
        if not math.isfinite(v):
            raise RuntimeError(f"Invalid inference output (non-finite): {v}")
        if reject_zero and v == 0.0:
            raise RuntimeError("Invalid inference output: zero value rejected for price topic")
        return v

    wallet_cfg = AlloraWalletConfig(mnemonic_file=mnemonic_file) if mnemonic_file else None
    net_cfg = _build_network(network, no_faucet)
    worker = AlloraWorker(run=run_fn, topic_id=topic_id, api_key=api_key, wallet=wallet_cfg, network=net_cfg, debug=debug)
    async for _ in worker.run():
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a managed Allora worker")
    parser.add_argument("--topic", type=int, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--mnemonic-file", default=None, help="Path to wallet key file (managed by WorkerManager)")
    parser.add_argument("--network", default="testnet", choices=["testnet", "mainnet"])
    parser.add_argument("--no-faucet", action="store_true", help="Skip SDK faucet checks (use when already funded)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reject-zero", action="store_true")
    args = parser.parse_args()

    api_key = _load_api_key(args.api_key)
    asyncio.run(_run(
        topic_id=args.topic,
        artifact_path=args.artifact,
        api_key=api_key,
        mnemonic_file=args.mnemonic_file,
        network=args.network,
        no_faucet=args.no_faucet,
        debug=args.debug,
        reject_zero=args.reject_zero,
    ))


if __name__ == "__main__":
    main()
