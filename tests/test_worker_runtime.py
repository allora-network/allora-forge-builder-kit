import pytest

from allora_forge_builder_kit import worker_runtime
from allora_forge_builder_kit.worker_runtime import (
    _TESTNET_FAUCET_URL,
    _artifact_expects_context,
    _build_network,
    _resolve_wallet_cfg,
    main,
)


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


def test_main_managed_without_forge_api_key_errors(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["worker_runtime", "--topic", "1", "--artifact", "x.pkl", "--custody", "managed"],
    )
    monkeypatch.delenv("FORGE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main()


def test_artifact_expects_context_legacy_nonce():
    def legacy(nonce):
        return 1.0

    assert _artifact_expects_context(legacy) is False


def test_artifact_expects_context_runcontext_by_name():
    def modern(ctx):
        return 1.0

    assert _artifact_expects_context(modern) is True


def test_artifact_expects_context_runcontext_by_annotation():
    def modern(x: "RunContext"):  # noqa: F821 - forward ref string is the point
        return 1.0

    assert _artifact_expects_context(modern) is True


def test_artifact_expects_context_multiarg_defaults_false():
    def two(a, b):
        return 1.0

    assert _artifact_expects_context(two) is False


def test_artifact_expects_context_uninspectable_defaults_false():
    assert _artifact_expects_context(object()) is False
