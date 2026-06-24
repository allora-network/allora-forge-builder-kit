"""Run an Allora worker from a pickled inference artifact.

Two distinct API keys are used here and must not be confused:

- ``ALLORA_API_KEY`` (``--api-key``): the Allora consumer/faucet key the kit uses for testnet
  faucet drips and topic queries (see ``_load_api_key``).
- ``FORGE_API_KEY``: the Forge backend key the SDK's ``AlloraWalletConfig.from_env()`` reads under
  ``--custody managed`` to provision and sign with a Privy-managed wallet.

They authenticate different backends; ``--custody managed`` consumes ``FORGE_API_KEY`` only.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import os
import warnings
from typing import TYPE_CHECKING, Callable, Literal

import cloudpickle

from allora_sdk.worker import AlloraWorker
from allora_sdk.rpc_client.config import AlloraNetworkConfig, AlloraWalletConfig

if TYPE_CHECKING:
    from allora_sdk.worker.context import RunContext


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


def _resolve_wallet_cfg(
    custody: Literal["local", "managed"], mnemonic_file: str | None
) -> AlloraWalletConfig | None:
    """Resolve the signing-wallet config for the chosen custody mode.

    Managed custody runs ``AlloraWalletConfig.from_env()``. With ``FORGE_API_KEY`` set and no
    ``FORGE_SIGNING_WALLET_ID``, the SDK returns a deferred managed config (no local key) and the
    worker get-or-creates a wallet bound to its topic at startup (ENGN-8646). The managed branch is
    taken before any ``PRIVATE_KEY`` / ``MNEMONIC`` is read, so ``FORGE_API_KEY`` always takes
    precedence — there is no silent local-key fallback. ``from_env()`` performs a blocking
    wallet-info fetch, so it is resolved here in sync code (called from ``main`` before the event
    loop starts) rather than inside the async worker.
    """
    if custody == "managed":
        return AlloraWalletConfig.from_env()
    return AlloraWalletConfig(mnemonic_file=mnemonic_file) if mnemonic_file else None


def _artifact_expects_context(raw_fn: Callable[..., object]) -> bool:
    """Return True when a pickled artifact follows the SDK's RunContext call contract.

    Legacy kit artifacts are written as ``fn(nonce: int)``; the current SDK invokes the inferer
    callback with a ``RunContext``. Default to the legacy nonce form and only pass the
    ``RunContext`` when the artifact's single positional parameter is annotated as — or named
    like — a context, preserving back-compat for the overwhelmingly common legacy artifacts.
    """
    try:
        params = [
            p
            for p in inspect.signature(raw_fn).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return False
    if len(params) != 1:
        return False
    annotation = "" if params[0].annotation is inspect.Parameter.empty else str(params[0].annotation)
    return "RunContext" in annotation or params[0].name in ("ctx", "context", "run_context")


async def _run(
    topic_id: int,
    artifact_path: str,
    api_key: str,
    wallet_cfg: AlloraWalletConfig | None = None,
    network: str = "testnet",
    no_faucet: bool = False,
    debug: bool = False,
    reject_zero: bool = False,
) -> None:
    """Load the pickled inference artifact and drive the worker submission loop.

    Artifact contract: the pickled callable is invoked as ``fn(nonce: int)`` and must return a
    finite numeric value. Artifacts written to the newer SDK contract (``fn(ctx)`` taking a
    ``RunContext``) are also supported; the call shape is detected once at load time.
    """
    with open(artifact_path, "rb") as f:
        raw_fn = cloudpickle.load(f)
    expects_context = _artifact_expects_context(raw_fn)

    def run_fn(ctx: RunContext) -> float:
        # Legacy artifacts take the integer nonce (raw_fn(ctx.nonce)); newer artifacts take the
        # RunContext itself. The call shape is resolved once above to avoid per-nonce introspection.
        value = raw_fn(ctx) if expects_context else raw_fn(ctx.nonce)
        try:
            v = float(value)
        except Exception as e:
            raise RuntimeError(f"Invalid inference output type: {value!r}") from e
        if not math.isfinite(v):
            raise RuntimeError(f"Invalid inference output (non-finite): {v}")
        if reject_zero and v == 0.0:
            raise RuntimeError("Invalid inference output: zero value rejected for price topic")
        return v

    net_cfg = _build_network(network, no_faucet)
    worker = AlloraWorker.inferer(
        run=run_fn,
        wallet=wallet_cfg,
        network=net_cfg,
        api_key=api_key,
        topic_id=topic_id,
        debug=debug,
    )
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
    parser.add_argument(
        "--custody",
        choices=["local", "managed"],
        default="local",
        help="Wallet custody: 'local' (self-custodial key file) or 'managed' (Privy-managed; "
        "provisions a wallet bound to the topic via FORGE_API_KEY, no local key)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reject-zero", action="store_true")
    args = parser.parse_args()

    if args.custody == "managed" and args.mnemonic_file:
        parser.error(
            "--mnemonic-file is incompatible with --custody managed; managed custody uses "
            "FORGE_API_KEY from the environment"
        )

    if args.custody == "managed" and not os.environ.get("FORGE_API_KEY"):
        parser.error(
            "--custody managed requires FORGE_API_KEY in the environment "
            "(the SDK provisions a topic-bound managed wallet from it)"
        )

    if args.custody == "managed" and not os.environ.get("FEE_GRANTER"):
        warnings.warn(
            "FEE_GRANTER is not set: a managed wallet holds no ALLO, so gasless submission needs "
            "a fee granter — transactions may fail with 'insufficient fees' without one.",
            stacklevel=2,
        )

    api_key = _load_api_key(args.api_key)
    wallet_cfg = _resolve_wallet_cfg(args.custody, args.mnemonic_file)
    asyncio.run(_run(
        topic_id=args.topic,
        artifact_path=args.artifact,
        api_key=api_key,
        wallet_cfg=wallet_cfg,
        network=args.network,
        no_faucet=args.no_faucet,
        debug=args.debug,
        reject_zero=args.reject_zero,
    ))


if __name__ == "__main__":
    main()
