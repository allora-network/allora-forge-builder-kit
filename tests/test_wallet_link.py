"""Tests for the wallet-link device-flow CLI (ENGN-8614)."""

from __future__ import annotations

import base64
import json

import pytest

from allora_forge_builder_kit.wallet_link import (
    _RequestError,
    _is_terminal_poll_error,
    _loads_json_object,
    _submit_rejection,
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


def test_discover_keys_skips_non_string_entries(tmp_path):
    # A tampered/malformed secrets file with non-string address/key_file must be skipped with a
    # clean result, not crash inside _checked_key_file on Path(<int>) (only json/OSError are caught).
    key_file = tmp_path / "w.key"
    key_file.write_text("mnemonic")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(
        json.dumps(
            {
                "bad_addr": {"address": 123, "key_file": str(key_file)},
                "bad_kf": {"address": "allo1bad", "key_file": 456},
                "good": {"address": "allo1good", "key_file": str(key_file)},
            }
        )
    )
    keys = discover_keys(str(secrets))
    assert set(keys) == {"allo1good"}


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


def test_submit_rejection_reports_rejected_signatures():
    msg = _submit_rejection({"rejected": [{"address": "allo1abc", "reason": "bad signature"}]})
    assert msg is not None
    assert "allo1abc" in msg
    assert "bad signature" in msg


def test_submit_rejection_reports_top_level_error():
    assert _submit_rejection({"error": "device code mismatch"}) is not None


def test_submit_rejection_none_when_no_rejections():
    assert _submit_rejection({"linked": []}) is None
    assert _submit_rejection({"rejected": []}) is None
    assert _submit_rejection({}) is None


def test_jsonposter_selects_https_proxy_from_env(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local:3128"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    assert poster._proxy == ("proxy.local", 3128)


def test_jsonposter_no_proxy_when_env_unset(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    assert poster._proxy is None


def test_jsonposter_respects_no_proxy_bypass(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local:3128"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: True)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    assert poster._proxy is None


def test_jsonposter_connect_tunnels_https_through_proxy(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local:3128"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com:8443")
    conn = poster._connect()
    try:
        # The socket targets the proxy; the CONNECT tunnel points at the real Forge host so TLS is
        # still validated against it.
        assert conn.host == "proxy.local"
        assert conn.port == 3128
        assert conn._tunnel_host == "forge.example.com"
        assert conn._tunnel_port == 8443
    finally:
        conn.close()


def test_jsonposter_authenticated_proxy_builds_proxy_authorization(monkeypatch):
    import base64

    from allora_forge_builder_kit import wallet_link

    # Userinfo is percent-encoded (p%40ss == "p@ss") to confirm it is URL-unquoted before base64.
    monkeypatch.setattr(
        wallet_link.urllib.request, "getproxies", lambda: {"https": "http://user:p%40ss@proxy.local:3128"}
    )
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")

    assert poster._proxy == ("proxy.local", 3128)  # host/port still parsed; creds kept separately
    expected = "Basic " + base64.b64encode(b"user:p@ss").decode("ascii")
    assert poster._proxy_auth == expected


def test_jsonposter_unauthenticated_proxy_has_no_auth(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local:3128"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")

    assert poster._proxy == ("proxy.local", 3128)
    assert poster._proxy_auth is None


def test_jsonposter_connect_sends_proxy_auth_on_tunnel(monkeypatch):
    import base64

    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(
        wallet_link.urllib.request, "getproxies", lambda: {"https": "http://user:pass@proxy.local:3128"}
    )
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com:8443")
    conn = poster._connect()
    try:
        expected = "Basic " + base64.b64encode(b"user:pass").decode("ascii")
        # The credentials ride the CONNECT request, not the tunneled origin request.
        assert conn._tunnel_headers.get("Proxy-Authorization") == expected
    finally:
        conn.close()


def test_jsonposter_unauthenticated_tunnel_has_no_proxy_auth(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local:3128"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com:8443")
    conn = poster._connect()
    try:
        assert "Proxy-Authorization" not in conn._tunnel_headers
    finally:
        conn.close()


def test_sign_challenge_address_mismatch():
    """A mnemonic whose address differs from the requested one is rejected."""
    pytest.importorskip("cosmpy")
    from cosmpy.mnemonic import generate_mnemonic

    from allora_forge_builder_kit.wallet_link import sign_challenge

    with pytest.raises(ValueError):
        sign_challenge(generate_mnemonic(), "allo1definitelynottheright", "msg")


def _setup_device_flow(monkeypatch, tmp_path, poll_result, addresses=("allo1aaa", "allo1bbb")):
    """Drive run_link's device flow against a stubbed server returning ``poll_result`` on poll."""
    from allora_forge_builder_kit import wallet_link

    key_file = tmp_path / "w.key"
    key_file.write_text("mnemonic")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(
        json.dumps({a: {"address": a, "key_file": str(key_file)} for a in addresses})
    )

    def fake_post_json(url, payload, timeout=15.0):
        if url.endswith("/device/start"):
            return {
                "device_code": "dev",
                "user_code": "USER",
                "verification_uri_complete": "https://forge.example/approve",
                "interval": 1,
                "expires_in": 5,
                "challenges": [{"address": a, "message": "m"} for a in addresses],
            }
        return {"status": "submitted"}

    class _FakePoller:
        def __init__(self, *a, **k):
            pass

        def post(self, url, payload):
            return poll_result

        def close(self):
            pass

    monkeypatch.setattr(wallet_link, "_post_json", fake_post_json)
    monkeypatch.setattr(wallet_link, "_JsonPoster", _FakePoller)
    monkeypatch.setattr(wallet_link, "sign_challenge", lambda *a: ("pub", "sig"))
    monkeypatch.setattr(wallet_link.webbrowser, "open", lambda url: False)
    monkeypatch.setattr(wallet_link.time, "sleep", lambda s: None)
    return str(secrets)


def test_run_link_partial_linked_set_is_failure(tmp_path, monkeypatch, capsys):
    # Approved but only a subset linked: must fail (exit 1) and name the dropped address, not
    # silently exit 0 (which would signal success to CI for a partial/failed link).
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(monkeypatch, tmp_path, {"status": "approved", "linked": ["allo1aaa"]})
    rc = wallet_link.run_link(forge_url="https://forge.example", secrets_path=secrets, open_browser=False)
    assert rc == 1
    assert "allo1bbb" in capsys.readouterr().err


def test_run_link_null_linked_does_not_crash(tmp_path, monkeypatch):
    # {"linked": null} previously made poll.get("linked", []) return None and crash len(None).
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(monkeypatch, tmp_path, {"status": "approved", "linked": None})
    rc = wallet_link.run_link(forge_url="https://forge.example", secrets_path=secrets, open_browser=False)
    assert rc == 1


def test_run_link_full_linked_set_succeeds(tmp_path, monkeypatch):
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(
        monkeypatch, tmp_path, {"status": "approved", "linked": ["allo1aaa", "allo1bbb"]}
    )
    rc = wallet_link.run_link(forge_url="https://forge.example", secrets_path=secrets, open_browser=False)
    assert rc == 0


def test_run_link_rejects_forge_url_with_path(capsys):
    # https://forge.allora.network/api/v1 would yield a double /api/v1 and a confusing 404;
    # reject it up front (before any network) with an actionable message.
    from allora_forge_builder_kit import wallet_link

    rc = wallet_link.run_link(forge_url="https://forge.example/api/v1", open_browser=False)
    assert rc == 1
    assert "path" in capsys.readouterr().err.lower()
