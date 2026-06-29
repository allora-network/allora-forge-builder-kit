"""Tests for the wallet-link device-flow CLI (ENGN-8614)."""

from __future__ import annotations

import base64
import json

import pytest

from allora_forge_builder_kit.wallet_link import (
    _RequestError,
    _is_terminal_poll_error,
    _loads_json_object,
    build_adr036_sign_doc,
    discover_keys,
)

# Byte-for-byte parity with the Go verifier's golden
# (forge-v2/backend/internal/service/adr036_test.go TestBuildADR036SignDoc_Golden).
GO_GOLDEN = (
    b'{"account_number":"0","chain_id":"","fee":{"amount":[],"gas":"0"},'
    b'"memo":"","msgs":[{"type":"sign/MsgSignData","value":{"data":'
    b'"aGVsbG8gd29ybGQ=","signer":"allo1xyz"}}],"sequence":"0"}'
)


def test_adr036_doc_matches_go_golden():
    assert build_adr036_sign_doc("allo1xyz", "hello world") == GO_GOLDEN


def test_discover_keys(tmp_path):
    key_file = tmp_path / "w.key"
    key_file.write_text("twelve word mnemonic goes here ...")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(
        json.dumps({"alias1": {"address": "allo1abc", "key_file": str(key_file)}})
    )

    keys = discover_keys(str(secrets))
    assert "allo1abc" in keys
    assert keys["allo1abc"]["alias"] == "alias1"
    assert keys["allo1abc"]["key_file"] == str(key_file)


def test_discover_keys_missing_file(tmp_path):
    assert discover_keys(str(tmp_path / "nope.json")) == {}


def test_sign_roundtrip_with_cosmpy():
    """sign_challenge must round-trip against the address derived from the key,
    and produce a 33-byte pubkey + 64-byte signature the Go verifier accepts."""
    pytest.importorskip("cosmpy")
    from cosmpy.aerial.wallet import LocalWallet
    from cosmpy.mnemonic import generate_mnemonic

    from allora_forge_builder_kit.wallet_link import sign_challenge

    mnemonic = generate_mnemonic()
    address = str(LocalWallet.from_mnemonic(mnemonic, "allo").address())

    pubkey_b64, signature_b64 = sign_challenge(mnemonic, address, "verify me")
    assert len(base64.b64decode(pubkey_b64)) == 33
    assert len(base64.b64decode(signature_b64)) == 64


def test_loads_json_object_accepts_object():
    assert _loads_json_object("https://forge.example", '{"device_code": "abc"}') == {"device_code": "abc"}


def test_loads_json_object_rejects_non_object_json():
    # A JSON array/scalar/null would crash callers with AttributeError on .get(); reject cleanly.
    for body in ("[1, 2, 3]", '"a string"', "null", "42"):
        with pytest.raises(SystemExit):
            _loads_json_object("https://forge.example", body)


def test_loads_json_object_rejects_non_json_body():
    # A truncated body or an HTML error page from an intermediary proxy must not raise an
    # unwrapped JSONDecodeError; it becomes a SystemExit the poll loop can treat as transient.
    with pytest.raises(SystemExit):
        _loads_json_object("https://forge.example", "<html>502 Bad Gateway</html>")


def test_is_terminal_poll_error_classifies_status():
    # Terminal 4xx (except 408/429) stops the flow; 5xx / 408 / 429 / network / malformed retry.
    assert _is_terminal_poll_error(_RequestError("x", status=404)) is True
    assert _is_terminal_poll_error(_RequestError("x", status=400)) is True
    assert _is_terminal_poll_error(_RequestError("x", status=403)) is True
    assert _is_terminal_poll_error(_RequestError("x", status=500)) is False
    assert _is_terminal_poll_error(_RequestError("x", status=503)) is False
    assert _is_terminal_poll_error(_RequestError("x", status=429)) is False
    assert _is_terminal_poll_error(_RequestError("x", status=408)) is False
    assert _is_terminal_poll_error(_RequestError("x")) is False  # network error: status is None
    assert _is_terminal_poll_error(SystemExit("malformed body")) is False  # no status attribute


def test_post_json_http_error_carries_status(monkeypatch):
    import io
    import urllib.error

    from allora_forge_builder_kit import wallet_link

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("https://forge.example", 404, "Not Found", {}, io.BytesIO(b"nope"))

    monkeypatch.setattr(wallet_link.urllib.request, "urlopen", _raise)
    with pytest.raises(_RequestError) as excinfo:
        wallet_link._post_json("https://forge.example", {})
    assert excinfo.value.status == 404


def test_sign_challenge_address_mismatch():
    """A mnemonic whose address differs from the requested one is rejected."""
    pytest.importorskip("cosmpy")
    from cosmpy.mnemonic import generate_mnemonic

    from allora_forge_builder_kit.wallet_link import sign_challenge

    with pytest.raises(ValueError):
        sign_challenge(generate_mnemonic(), "allo1definitelynottheright", "msg")
