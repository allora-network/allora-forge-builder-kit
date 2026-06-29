import pytest

from allora_forge_builder_kit import worker_runtime
from allora_forge_builder_kit.worker_runtime import (
    _TESTNET_FAUCET_URL,
    _ArtifactCaller,
    _build_network,
    _resolve_wallet_cfg,
    main,
)


class _FakeCtx:
    """Minimal stand-in for the SDK RunContext: only exposes the integer nonce."""

    def __init__(self, nonce: int):
        self.nonce = nonce


def test_build_network_testnet_overrides_faucet_url():
    cfg = _build_network("testnet", no_faucet=False)
    assert cfg.faucet_url == _TESTNET_FAUCET_URL


def test_build_network_mainnet_does_not_set_faucet_url():
    cfg = _build_network("mainnet", no_faucet=False)
    assert cfg.faucet_url != _TESTNET_FAUCET_URL


def test_build_network_no_faucet_clears_url():
    for network in ("testnet", "mainnet"):
        cfg = _build_network(network, no_faucet=True)
        assert cfg.faucet_url is None


class _FakeWalletConfig:
    """Stand-in for AlloraWalletConfig that records how it is constructed/invoked."""

    last_init_kwargs: dict | None = None
    from_env_calls: int = 0
    from_env_sentinel = object()

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    @classmethod
    def from_env(cls):
        cls.from_env_calls += 1
        return cls.from_env_sentinel


@pytest.fixture
def fake_wallet_config(monkeypatch):
    _FakeWalletConfig.last_init_kwargs = None
    _FakeWalletConfig.from_env_calls = 0
    monkeypatch.setattr(worker_runtime, "AlloraWalletConfig", _FakeWalletConfig)
    return _FakeWalletConfig


def test_resolve_wallet_cfg_managed_calls_from_env(fake_wallet_config):
    cfg = _resolve_wallet_cfg("managed", None)
    assert cfg is fake_wallet_config.from_env_sentinel
    assert fake_wallet_config.from_env_calls == 1
    # Managed routing must not construct a local AlloraWalletConfig(...).
    assert fake_wallet_config.last_init_kwargs is None


def test_resolve_wallet_cfg_managed_ignores_mnemonic_file(fake_wallet_config):
    cfg = _resolve_wallet_cfg("managed", "/tmp/key.txt")
    assert cfg is fake_wallet_config.from_env_sentinel
    assert fake_wallet_config.last_init_kwargs is None


def test_resolve_wallet_cfg_local_builds_from_mnemonic_file(fake_wallet_config):
    _resolve_wallet_cfg("local", "/tmp/key.txt")
    assert fake_wallet_config.from_env_calls == 0
    assert fake_wallet_config.last_init_kwargs == {"mnemonic_file": "/tmp/key.txt"}


def test_resolve_wallet_cfg_local_without_mnemonic_is_none(fake_wallet_config):
    assert _resolve_wallet_cfg("local", None) is None
    assert fake_wallet_config.from_env_calls == 0


def test_main_managed_with_mnemonic_file_errors(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "worker_runtime",
            "--topic", "1",
            "--artifact", "x.pkl",
            "--custody", "managed",
            "--mnemonic-file", "k.txt",
        ],
    )
    monkeypatch.setenv("FORGE_API_KEY", "fk")
    with pytest.raises(SystemExit):
        main()


def test_main_local_without_mnemonic_file_errors(monkeypatch):
    # Symmetric with the managed branch: local custody requires --mnemonic-file, otherwise the SDK
    # would silently generate a throwaway wallet. Fail at arg-parse time instead.
    monkeypatch.setattr(
        "sys.argv",
        ["worker_runtime", "--topic", "1", "--artifact", "x.pkl", "--custody", "local"],
    )
    with pytest.raises(SystemExit):
        main()


def test_main_managed_without_forge_api_key_errors(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["worker_runtime", "--topic", "1", "--artifact", "x.pkl", "--custody", "managed"],
    )
    monkeypatch.delenv("FORGE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main()


def test_artifact_caller_legacy_nonce_named_nonce():
    def legacy(nonce):
        return float(nonce)

    assert _ArtifactCaller(legacy)(_FakeCtx(7)) == 7.0


def test_artifact_caller_legacy_nonce_named_ctx_is_not_misrouted():
    # A legacy int-taking fn whose parameter is named `ctx` must still receive the nonce, not the
    # RunContext object — the old name heuristic mis-routed exactly this case.
    seen = {}

    def legacy(ctx):
        seen["arg"] = ctx
        return float(ctx) + 1  # raises TypeError if handed a RunContext object

    assert _ArtifactCaller(legacy)(_FakeCtx(4)) == 5.0
    assert seen["arg"] == 4


def test_artifact_caller_modern_runcontext_named_run_ctx():
    # A modern fn that reads ctx.nonce, with a parameter name outside the old heuristic set.
    def modern(run_ctx):
        return float(run_ctx.nonce * 2)

    assert _ArtifactCaller(modern)(_FakeCtx(3)) == 6.0


def test_artifact_caller_annotated_modern_not_demoted_by_body_typeerror():
    # cubic: a RunContext-annotated artifact whose body raises TypeError on its first call must not
    # be misclassified as legacy (which would then pass an int where a RunContext is expected). The
    # annotation is authoritative, so the shape stays 'context' and the body error propagates.
    class RunContext:
        def __init__(self, nonce):
            self.nonce = nonce

    calls = []

    def modern(ctx: RunContext):
        calls.append(ctx)
        if len(calls) == 1:
            raise TypeError("transient error from the artifact body, not a signature mismatch")
        return float(ctx.nonce)

    caller = _ArtifactCaller(modern)
    with pytest.raises(TypeError):
        caller(_FakeCtx(1))
    # Still routed via the context form; the ctx object is passed on the retry, never an int nonce.
    assert caller(_FakeCtx(2)) == 2.0
    assert all(not isinstance(c, int) for c in calls)


def test_artifact_caller_unannotated_modern_body_typeerror_not_demoted():
    # An *unannotated* single-param modern artifact whose body raises TypeError on its first call
    # must not be permanently demoted to the legacy nonce form: the int fallback fails too, so the
    # original TypeError propagates and the shape stays unresolved (re-probed) instead of caching
    # the wrong legacy contract for the rest of the worker's life.
    calls = []

    def modern(ctx):  # unannotated -> probed at call time
        calls.append(ctx)
        if len(calls) == 1:
            raise TypeError("transient body error, not a signature mismatch")
        return float(ctx.nonce)

    caller = _ArtifactCaller(modern)
    with pytest.raises(TypeError, match="transient body error"):
        caller(_FakeCtx(1))
    # Not demoted: shape stays unresolved, so the next call probes the context form and succeeds.
    assert caller._expects_context is None
    assert caller(_FakeCtx(2)) == 2.0
    assert caller._expects_context is True


def test_artifact_caller_caches_modern_shape_without_reprobe():
    calls = []

    def modern(ctx):
        calls.append(ctx.nonce)
        return float(ctx.nonce)

    call = _ArtifactCaller(modern)
    call(_FakeCtx(1))
    call(_FakeCtx(2))
    # The RunContext form succeeded on the first probe, so the shape is cached and the second
    # call goes straight through — no fallback, each nonce seen exactly once.
    assert calls == [1, 2]
