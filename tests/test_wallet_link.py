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
    SecretsLoadError,
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


def test_discover_keys_raises_on_corrupt_json(tmp_path):
    # A present-but-corrupt secrets file is distinct from absent: raise SecretsLoadError rather
    # than return {} (which would mislead the caller into "no worker keys, create one").
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text("{not valid json")
    with pytest.raises(SecretsLoadError):
        discover_keys(str(secrets))


def test_discover_keys_raises_on_non_object_root(tmp_path):
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SecretsLoadError):
        discover_keys(str(secrets))


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

    monkeypatch.setattr(wallet_link._OPENER, "open", _raise)
    with pytest.raises(_RequestError) as excinfo:
        wallet_link._post_json("https://forge.example", {})
    assert excinfo.value.status == 404


def test_post_json_opener_refuses_redirects():
    # The start/submit opener must not follow 3xx: redirect_request returning None makes urllib
    # raise HTTPError instead of re-POSTing the signed body to the redirect target.
    from allora_forge_builder_kit import wallet_link

    handler = wallet_link._NoRedirectHandler()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/") is None
    assert any(isinstance(h, wallet_link._NoRedirectHandler) for h in wallet_link._OPENER.handlers)


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


def test_jsonposter_warns_on_proxy_without_explicit_port(monkeypatch, capsys):
    from allora_forge_builder_kit import wallet_link

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {"https": "http://proxy.local"})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    assert poster._proxy == ("proxy.local", None)  # port stays None (http.client defaults it)
    err = capsys.readouterr().err
    assert "no explicit port" in err and "443" in err


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


def test_run_link_rejects_verification_url_on_different_port(tmp_path, monkeypatch, capsys):
    # A compromised server returning the approval URL on a different port of the same host must be
    # refused: the port is part of the origin the CLI pins.
    from allora_forge_builder_kit import wallet_link

    key_file = tmp_path / "w.key"
    key_file.write_text("mnemonic")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(json.dumps({"allo1aaa": {"address": "allo1aaa", "key_file": str(key_file)}}))

    def fake_post_json(url, payload, timeout=15.0):
        return {
            "device_code": "dev",
            "user_code": "USER",
            "verification_uri_complete": "https://forge.example:8443/approve",
            "interval": 1,
            "expires_in": 5,
            "challenges": [{"address": "allo1aaa", "message": "m"}],
        }

    monkeypatch.setattr(wallet_link, "_post_json", fake_post_json)
    rc = wallet_link.run_link(forge_url="https://forge.example", secrets_path=str(secrets), open_browser=False)
    assert rc == 1
    assert "untrusted verification URL" in capsys.readouterr().err


def test_jsonposter_closes_connection_on_http_error(monkeypatch):
    # A >= 400 response raises _RequestError (a BaseException the keep-alive except clause won't
    # catch), so the connection must be closed first — otherwise the next poll reuses a connection
    # whose body may be undrained and defeats keep-alive.
    from allora_forge_builder_kit import wallet_link

    class _FakeResp:
        status = 403

        def read(self, n):
            return b"denied"

    class _FakeConn:
        def __init__(self):
            self.closed = False

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return _FakeResp()

        def close(self):
            self.closed = True

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    fake = _FakeConn()
    poster._conn = fake
    with pytest.raises(wallet_link._RequestError) as ei:
        poster.post("https://forge.example.com/poll", {})
    assert ei.value.status == 403
    assert fake.closed is True       # the dirty connection was closed
    assert poster._conn is None      # so the next poll reconnects fresh


# --- Security-hardening regression tests (ENGN-8614 review round) --------------------------------


def test_printable_strips_zero_width_and_bidi_chars():
    # _printable must drop Unicode Cf/control codepoints (zero-width, BOM, bidi controls, ESC) so a
    # server can't hide invisible/lookalike characters inside an address or code shown to the user.
    from allora_forge_builder_kit.wallet_link import _printable

    dangerous = "allo1abc\u200b\u200c\u200d\u202e\u2066\ufeff\x1b[2J"
    out = _printable(dangerous)
    assert out == "allo1abc[2J"  # visible text kept; ESC + every Cf char removed
    for cp in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C,
               0x202D, 0x202E, 0x2060, 0x2066, 0x2067, 0x2068, 0x2069, 0xFEFF):
        assert chr(cp) not in out


def test_loads_json_object_sanitizes_non_json_error():
    # The non-JSON error path must run the raw body through _printable before it reaches stderr.
    with pytest.raises(SystemExit) as ei:
        _loads_json_object("https://forge.example", "\x1b[2Jmalicious\u200b")
    msg = str(ei.value)
    assert "\x1b" not in msg and "\u200b" not in msg
    assert "malicious" in msg


def test_is_terminal_poll_error_treats_3xx_as_terminal():
    # A 3xx on the poll path is a misconfiguration (e.g. http->https redirect) that never resolves
    # by retrying, so it must be classified terminal alongside 4xx.
    assert _is_terminal_poll_error(_RequestError("x", status=301)) is True
    assert _is_terminal_poll_error(_RequestError("x", status=302)) is True
    assert _is_terminal_poll_error(_RequestError("x", status=308)) is True


def test_build_adr036_sign_doc_rejects_non_alphanumeric_signer():
    # The injection guard must reject anything outside [a-z0-9] so a signer can't corrupt the JSON.
    for bad in ("allo1ABC", 'allo1"', "allo1\\", "allo1;rm", "allo1 x", "allo1\x1b"):
        with pytest.raises(ValueError):
            build_adr036_sign_doc(bad, "msg")


def test_post_json_retrying_retries_transient_then_succeeds(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    calls = {"n": 0}

    def flaky(url, payload, timeout=15.0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise wallet_link._RequestError("temporary", status=503)  # transient -> retry
        return {"status": "submitted"}

    monkeypatch.setattr(wallet_link, "_post_json", flaky)
    monkeypatch.setattr(wallet_link.time, "sleep", lambda s: None)
    out = wallet_link._post_json_retrying("https://forge.example/submit", {}, attempts=3, backoff=0)
    assert out == {"status": "submitted"}
    assert calls["n"] == 3


def test_post_json_retrying_reraises_terminal_immediately(monkeypatch):
    from allora_forge_builder_kit import wallet_link

    calls = {"n": 0}

    def terminal(url, payload, timeout=15.0):
        calls["n"] += 1
        raise wallet_link._RequestError("bad request", status=400)  # terminal -> no retry

    monkeypatch.setattr(wallet_link, "_post_json", terminal)
    monkeypatch.setattr(wallet_link.time, "sleep", lambda s: None)
    with pytest.raises(wallet_link._RequestError):
        wallet_link._post_json_retrying("https://forge.example/submit", {}, attempts=3, backoff=0)
    assert calls["n"] == 1


def test_jsonposter_treats_3xx_as_error(monkeypatch):
    # A 3xx response must raise _RequestError (not fall through to _loads_json_object and spin).
    from allora_forge_builder_kit import wallet_link

    class _FakeResp:
        status = 302

        def read(self, n):
            return b"redirecting"

    class _FakeConn:
        def request(self, *a, **k):
            pass

        def getresponse(self):
            return _FakeResp()

        def close(self):
            pass

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    poster._conn = _FakeConn()
    with pytest.raises(wallet_link._RequestError) as ei:
        poster.post("https://forge.example.com/poll", {})
    assert ei.value.status == 302


def test_jsonposter_closes_connection_on_malformed_200(monkeypatch):
    # A non-JSON 200 body raises SystemExit; the connection must be closed so the next poll
    # reconnects instead of reusing a dirty socket (matches the >= 400 path).
    from allora_forge_builder_kit import wallet_link

    class _FakeResp:
        status = 200

        def read(self, n):
            return b"<html>not json</html>"

    class _FakeConn:
        def __init__(self):
            self.closed = False

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return _FakeResp()

        def close(self):
            self.closed = True

    monkeypatch.setattr(wallet_link.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(wallet_link.urllib.request, "proxy_bypass", lambda host: False)
    poster = wallet_link._JsonPoster("https://forge.example.com")
    fake = _FakeConn()
    poster._conn = fake
    with pytest.raises(SystemExit):
        poster.post("https://forge.example.com/poll", {})
    assert fake.closed is True
    assert poster._conn is None


def test_run_link_sign_failure_does_not_leak_mnemonic(tmp_path, monkeypatch, capsys):
    # A signing exception whose message embeds mnemonic words must NOT reach stderr; only the
    # exception type is surfaced.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(
        monkeypatch, tmp_path, {"status": "approved", "linked": []}, addresses=("allo1aaa",)
    )
    secret_words = "abandon ability able about above absent absorb"

    def boom(*a, **k):
        raise RuntimeError(f"cosmpy internal error: {secret_words}")

    monkeypatch.setattr(wallet_link, "sign_challenge", boom)
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert secret_words not in err
    assert "abandon" not in err
    assert "RuntimeError" in err  # type is safe to show


def test_run_link_wallet_sign_error_message_is_shown(tmp_path, monkeypatch, capsys):
    # WalletSignError carries a key-material-free message (address mismatch) and IS shown verbatim.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(
        monkeypatch, tmp_path, {"status": "approved"}, addresses=("allo1aaa",)
    )

    def mismatch(*a, **k):
        raise wallet_link.WalletSignError(
            "key derives address allo1zzz, which does not match requested allo1aaa"
        )

    monkeypatch.setattr(wallet_link, "sign_challenge", mismatch)
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "allo1zzz" in err


def test_run_link_empty_key_file_is_clear_error(tmp_path, monkeypatch, capsys):
    # A whitespace-only key file must fail with a clear "empty" message before cosmpy is called.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(
        monkeypatch, tmp_path, {"status": "approved"}, addresses=("allo1aaa",)
    )
    (tmp_path / "w.key").write_text("   \n\t")
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "empty" in err.lower()


def test_run_link_deduplicates_addresses(tmp_path, monkeypatch):
    # Repeated --address X --address X must collapse to a single challenge + signature.
    from allora_forge_builder_kit import wallet_link

    key_file = tmp_path / "w.key"
    key_file.write_text("mnemonic")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(json.dumps({"allo1aaa": {"address": "allo1aaa", "key_file": str(key_file)}}))
    captured = {}

    def fake_post_json(url, payload, timeout=15.0):
        if url.endswith("/device/start"):
            captured["start_addresses"] = list(payload["addresses"])
            return {
                "device_code": "dev",
                "user_code": "U",
                "verification_uri_complete": "https://forge.example/approve",
                "interval": 1,
                "expires_in": 5,
                "challenges": [{"address": "allo1aaa", "message": "m"}],
            }
        captured["submit_sigs"] = list(payload["signatures"])
        return {"status": "submitted"}

    class _P:
        def __init__(self, *a, **k):
            pass

        def post(self, u, p):
            return {"status": "approved", "linked": ["allo1aaa"]}

        def close(self):
            pass

    monkeypatch.setattr(wallet_link, "_post_json", fake_post_json)
    monkeypatch.setattr(wallet_link, "_JsonPoster", _P)
    monkeypatch.setattr(wallet_link, "sign_challenge", lambda *a: ("pub", "sig"))
    monkeypatch.setattr(wallet_link.webbrowser, "open", lambda u: False)
    monkeypatch.setattr(wallet_link.time, "sleep", lambda s: None)
    rc = wallet_link.run_link(
        forge_url="https://forge.example",
        secrets_path=str(secrets),
        addresses=["allo1aaa", "allo1aaa"],
        open_browser=False,
    )
    assert rc == 0
    assert captured["start_addresses"] == ["allo1aaa"]
    assert len(captured["submit_sigs"]) == 1


def test_run_link_empty_address_list_fails_closed(tmp_path, monkeypatch, capsys):
    # addresses=[] must link nothing and never hit the network (not fall back to linking all).
    from allora_forge_builder_kit import wallet_link

    key_file = tmp_path / "w.key"
    key_file.write_text("mnemonic")
    secrets = tmp_path / "worker_secrets.json"
    secrets.write_text(
        json.dumps(
            {
                "allo1aaa": {"address": "allo1aaa", "key_file": str(key_file)},
                "allo1bbb": {"address": "allo1bbb", "key_file": str(key_file)},
            }
        )
    )
    called = {"start": False}

    def fake_post_json(url, payload, timeout=15.0):
        called["start"] = True
        return {}

    monkeypatch.setattr(wallet_link, "_post_json", fake_post_json)
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=str(secrets), addresses=[], open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "no addresses" in err.lower()
    assert called["start"] is False


def test_run_link_unknown_poll_status_gives_up(tmp_path, monkeypatch, capsys):
    # An unrecognized poll status must bail (bounded) instead of polling silently to the deadline.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(monkeypatch, tmp_path, {"status": "wat"}, addresses=("allo1aaa",))
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "unrecognized poll status" in err


def test_run_link_persistent_poll_error_gives_up(tmp_path, monkeypatch, capsys):
    # A consistently malformed/transient poll response must give up after the consecutive bound
    # rather than retry for the full deadline.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(monkeypatch, tmp_path, None, addresses=("allo1aaa",))

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def post(self, u, p):
            raise SystemExit("malformed body")  # no status -> transient

        def close(self):
            pass

    monkeypatch.setattr(wallet_link, "_JsonPoster", _Boom)
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "consecutive poll errors" in err


def test_run_link_authorization_pending_then_approved(tmp_path, monkeypatch):
    # The forge-v2 pending status ("authorization_pending") must keep polling, then succeed.
    from allora_forge_builder_kit import wallet_link

    secrets = _setup_device_flow(monkeypatch, tmp_path, None, addresses=("allo1aaa",))

    class _Pending:
        def __init__(self, *a, **k):
            self.n = 0

        def post(self, u, p):
            self.n += 1
            if self.n < 3:
                return {"status": "authorization_pending"}
            return {"status": "approved", "linked": ["allo1aaa"]}

        def close(self):
            pass

    monkeypatch.setattr(wallet_link, "_JsonPoster", _Pending)
    rc = wallet_link.run_link(
        forge_url="https://forge.example", secrets_path=secrets, open_browser=False
    )
    assert rc == 0
