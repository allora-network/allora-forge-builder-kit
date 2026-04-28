from allora_forge_builder_kit.worker_runtime import _build_network, _TESTNET_FAUCET_URL


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
