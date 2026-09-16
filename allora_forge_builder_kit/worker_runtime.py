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
from allora_sdk.worker.context import RunContext

if TYPE_CHECKING:
    from allora_sdk.worker.context import RunContext


def _load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Strip the env value and treat a whitespace-only string as absent, mirroring the file
    # fallbacks below and WorkerManager._allora_api_key_present: otherwise the subprocess would
    # accept a garbage key the manager precheck rejects, resolving different effective credentials.
    env = os.environ.get("ALLORA_API_KEY", "").strip()
    if env:
        return env
    for path in ("notebooks/.allora_api_key", ".allora_api_key"):
        if os.path.exists(path):
            with open(path, "r") as f:
                key = f.read().strip()
                if key:
                    return key
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


def _validate_managed_env() -> None:
    """Warn about managed-custody env gaps the SDK would otherwise paper over silently.

    Lives on the wallet-config seam rather than only in ``main`` so a programmatic caller of
    ``_resolve_wallet_cfg('managed', ...)`` (a notebook, test harness, or other entry point) gets
    the same diagnostics as the CLI.
    """
    if not os.environ.get("FORGE_BACKEND_URL", "").strip():
        # WorkerManager-spawned managed workers always get FORGE_BACKEND_URL injected, but a direct
        # entry point may not. The SDK's AlloraWalletConfig.from_env() silently defaults an unset
        # URL to the public production backend, so warn loudly to avoid signing against prod when a
        # staging / self-hosted instance was intended.
        warnings.warn(
            "FORGE_BACKEND_URL is not set: managed custody will default to the public production "
            "backend (https://forge.allora.network); set it explicitly to target a staging or "
            "self-hosted Forge instance.",
            stacklevel=2,
        )
    # FORGE_MASTER_GRANTER_ADDRESS is the canonical fee-granter env var across the Allora SDKs
    # (allora-sdk-py/-go/-ts); FEE_GRANTER is the deprecated alias the SDK still accepts. Check the
    # canonical name first so an operator who set it correctly doesn't see a spurious warning.
    if not (
        os.environ.get("FORGE_MASTER_GRANTER_ADDRESS", "").strip()
        or os.environ.get("FEE_GRANTER", "").strip()
    ):
        warnings.warn(
            "No fee granter set (FORGE_MASTER_GRANTER_ADDRESS, or the deprecated FEE_GRANTER): a "
            "managed wallet holds no ALLO, so gasless submission needs a fee granter — transactions "
            "may fail with 'insufficient fees' without one.",
            stacklevel=2,
        )


def _resolve_wallet_cfg(
    custody: Literal["local", "managed"], mnemonic_file: str | None
) -> AlloraWalletConfig | None:
    """Resolve the signing-wallet config for the chosen custody mode.

    Managed custody validates the managed-custody env (:func:`_validate_managed_env`) and runs
    ``AlloraWalletConfig.from_env()``. Two managed entry points feed this, and the SDK branch differs
    between them:

    * Direct CLI (notebook / advanced use): only ``FORGE_API_KEY`` is set, no
      ``FORGE_SIGNING_WALLET_ID``. The SDK returns a deferred managed config (no local key) and the
      worker get-or-creates a wallet bound to its topic at startup (ENGN-8646).
    * WorkerManager-spawned: the manager injects ``FORGE_API_KEY`` **and** pins
      ``FORGE_SIGNING_WALLET_ID`` to the already-provisioned wallet, so ``from_env()`` takes the
      immediate ``make_remote_wallet`` fetch branch for that exact wallet rather than the deferred
      get-or-create.

    Either way the managed branch is taken before any ``PRIVATE_KEY`` / ``MNEMONIC`` is read, so
    ``FORGE_API_KEY`` always takes precedence — there is no silent local-key fallback. ``from_env()``
    performs a blocking wallet-info fetch, so it is resolved here in sync code (called from ``main``
    before the event loop starts) rather than inside the async worker.
    """
    if custody == "managed":
        _validate_managed_env()
        return AlloraWalletConfig.from_env()
    return AlloraWalletConfig(mnemonic_file=mnemonic_file) if mnemonic_file else None


def _annotated_context_shape(raw_fn: Callable[..., object]) -> bool | None:
    """Resolve the artifact call shape from its signature annotation, without invoking it.

    Returns True when the single positional parameter is annotated as a ``RunContext`` (modern
    form), False when it carries any other explicit annotation (legacy ``fn(nonce: int)`` form), or
    None when the shape cannot be decided from the signature alone — unannotated, zero/many
    positional params, or an un-introspectable builtin — and must be probed at call time.
    """
    try:
        params = [
            p
            for p in inspect.signature(raw_fn).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return None
    if len(params) != 1 or params[0].annotation is inspect.Parameter.empty:
        return None
    # Match the unqualified type name exactly (covers both the RunContext class object and a
    # string / forward-ref annotation) so a distinct type whose name merely *contains* "RunContext"
    # — e.g. MockRunContext, NotRunContext — is not misrouted to the modern fn(ctx) form.
    annotation = params[0].annotation
    name = getattr(annotation, "__name__", None) or str(annotation)
    return name.rsplit(".", 1)[-1] == "RunContext"


class _ArtifactCaller:
    """Resolve and cache how a pickled inference artifact wants to be invoked.

    The current SDK calls the inferer callback with a ``RunContext``; legacy kit artifacts are
    written as ``fn(nonce: int)``. The shape is taken from the parameter's annotation when one is
    present — a ``RunContext`` annotation is the modern form, any other annotation the legacy form —
    so a ``TypeError`` raised inside a modern artifact's own body is never misread as a legacy
    signature (the failure mode of probing by execution alone). Only an *unannotated* single-param
    artifact is probed at call time: try the ``RunContext`` form (a legacy int-taking function
    raises ``TypeError`` when handed a ``RunContext``), then fall back to the nonce form. The
    legacy shape is cached only once the nonce fallback actually succeeds; if that fallback also
    raises (e.g. a modern artifact whose own body raised the ``TypeError``), the original error
    propagates and the shape stays unresolved, so a transient failure can't permanently mis-route a
    context-taking artifact to the int-nonce form for the rest of the worker's life.
    """

    def __init__(self, raw_fn: Callable[..., object]) -> None:
        self._raw_fn = raw_fn
        self._expects_context: bool | None = _annotated_context_shape(raw_fn)

    def __call__(self, ctx: RunContext) -> object:
        if self._expects_context is True:
            return self._raw_fn(ctx)
        if self._expects_context is False:
            return self._raw_fn(ctx.nonce)
        try:
            value = self._raw_fn(ctx)
        except TypeError as original:
            # Only demote to the legacy nonce form if the fallback succeeds. If it raises too, the
            # TypeError came from the modern artifact's body, not a signature mismatch: re-raise it
            # and leave the shape unresolved rather than caching the wrong (legacy) contract.
            try:
                value = self._raw_fn(ctx.nonce)
            except Exception:
                raise original
            self._expects_context = False
            return value
        self._expects_context = True
        return value


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
    ``RunContext``) are also supported; the call shape is probed and cached on the first invocation.
    """
    with open(artifact_path, "rb") as f:
        raw_fn = cloudpickle.load(f)
    call_artifact = _ArtifactCaller(raw_fn)

    def run_fn(ctx: RunContext) -> float:
        value = call_artifact(ctx)
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
    parser.add_argument("--network", default=os.environ.get("ALLORA_NETWORK", "testnet"), choices=["testnet", "mainnet"])
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

    if args.custody == "local" and not args.mnemonic_file:
        # Symmetric with the managed branch's loud FORGE_API_KEY check below. Without a key file the
        # SDK falls back to its interactive ".allora_key" flow — generating and persisting a fresh
        # throwaway wallet (or blocking on a prompt) inside a headless subprocess — instead of
        # signing with the intended worker key, so fail at parse time with an actionable message.
        parser.error(
            "--custody local requires --mnemonic-file pointing at a worker key file; "
            "use --custody managed for backend-provisioned wallets"
        )

    if args.custody == "managed" and args.mnemonic_file:
        parser.error(
            "--mnemonic-file is incompatible with --custody managed; managed custody uses "
            "FORGE_API_KEY from the environment"
        )

    if args.custody == "managed" and not os.environ.get("FORGE_API_KEY", "").strip():
        parser.error(
            "--custody managed requires FORGE_API_KEY in the environment "
            "(the SDK provisions a topic-bound managed wallet from it)"
        )

    api_key = _load_api_key(args.api_key)
    os.environ["ALLORA_API_KEY"] = api_key  # artifacts read environ at predict time
    # _resolve_wallet_cfg validates the managed-custody env (FORGE_BACKEND_URL default-to-prod and
    # fee-granter warnings) on the seam, so those diagnostics fire for any caller, not just the CLI.
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
