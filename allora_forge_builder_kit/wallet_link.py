"""Link locally-managed worker wallets to a logged-in Allora Forge account.

This is the default, ``gh auth login``-style flow for ENGN-8614: the CLI signs an
ADR-036 challenge with the worker key on disk, opens the browser, the logged-in
user approves, and the CLI polls to completion. The mnemonic never leaves the dev
kit. Works headless too (it prints the URL + code to open on any device).

Usage::

    python -m allora_forge_builder_kit.wallet_link --forge-url https://forge.allora.network
    # or a subset of addresses / a single one:
    python -m allora_forge_builder_kit.wallet_link --address allo1... --address allo1...

The signature contract (canonical amino ``sign/MsgSignData`` doc, sorted keys, no
whitespace) MUST stay byte-for-byte identical to the Go verifier in
``forge-v2/backend/internal/service/adr036.go`` and to Keplr's ``signArbitrary``.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import unquote, urlparse, urlsplit

DEFAULT_FORGE_URL = "https://forge.allora.network"
DEFAULT_SECRETS_PATH = "worker_secrets.json"
# Ceiling on how long the device-flow poll loop waits for browser approval. The client honors the
# server-advertised expires_in but clamps it to this bound (with a 1s floor). Sized to the typical
# RFC 8628 device-authorization session window (900-1800s); a lower ceiling would abort sessions the
# server still considers live (e.g. an approval delayed by MFA or a device switch).
_POLL_TIMEOUT_SECONDS = 1800
_MAX_RESPONSE_BYTES = 512 * 1024
# Loopback hosts treated as safe for plaintext HTTP / verification-URL origin pinning. Includes the
# IPv6 loopback ::1 (urlparse('http://[::1]/').hostname == '::1', no brackets) so a local Forge bound
# to [::1] on a dual-stack host doesn't require --insecure.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


# A single discovered worker key entry from worker_secrets.json.
class _KeyEntry(TypedDict):
    alias: str
    key_file: str


def build_adr036_sign_doc(signer: str, message: str) -> bytes:
    """Return the canonical Cosmos ADR-036 amino StdSignDoc bytes.

    Must match the Go verifier and Keplr byte-for-byte: keys sorted alphabetically
    at every level, no whitespace, ``data`` = standard-base64 of the raw message.
    """
    # signer is concatenated unescaped; restrict it to the bech32 grammar so a
    # stray quote/backslash/control byte can't corrupt or inject into the JSON.
    if not re.fullmatch(r"[a-z0-9]+", signer):
        raise ValueError(f"invalid bech32 signer: {signer!r}")
    data = base64.standard_b64encode(message.encode("utf-8")).decode("ascii")
    return (
        '{"account_number":"0","chain_id":"","fee":{"amount":[],"gas":"0"},'
        '"memo":"","msgs":[{"type":"sign/MsgSignData","value":{"data":"'
        + data
        + '","signer":"'
        + signer
        + '"}}],"sequence":"0"}'
    ).encode("utf-8")


class WalletSignError(ValueError):
    """A signing failure whose message is guaranteed free of key material.

    Subclasses ``ValueError`` for backward compatibility. Raised only for failures whose message is
    known to contain no mnemonic bytes (e.g. a derived-address mismatch), so ``run_link`` can echo
    it verbatim while treating every *other* signing exception as opaque (type name only) to keep
    private key material out of stderr and logs.
    """


def sign_challenge(mnemonic: str, address: str, message: str) -> tuple[str, str]:
    """Sign an ADR-036 challenge with the mnemonic's key.

    Returns ``(pubkey_b64, signature_b64)`` where pubkey is the 33-byte compressed
    secp256k1 key and signature is the 64-byte compact (r||s) form, both
    standard-base64. cosmpy's ``PrivateKey.sign`` signs ``sha256(doc)`` and returns
    the canonical 64-byte signature, matching the server's verification.
    """
    try:
        from cosmpy.aerial.wallet import LocalWallet
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "cosmpy is required to sign. Install the wallet-link extra "
            "(pip install 'allora-forge-builder-kit[wallet-link]') or cosmpy==0.11.1."
        ) from exc

    wallet = LocalWallet.from_mnemonic(mnemonic, "allo")
    derived = str(wallet.address())
    if derived != address:
        # WalletSignError (not a bare ValueError): its message holds only bech32 addresses, so the
        # caller may show it verbatim, unlike a cosmpy exception that could embed mnemonic bytes.
        raise WalletSignError(
            f"key derives address {derived}, which does not match requested {address}"
        )

    signer = wallet.signer()
    doc = build_adr036_sign_doc(address, message)
    signature = signer.sign(doc)
    pubkey_b64 = base64.standard_b64encode(signer.public_key.public_key_bytes).decode("ascii")
    signature_b64 = base64.standard_b64encode(signature).decode("ascii")
    return pubkey_b64, signature_b64


def _checked_key_file(base: str, address: str, key_file: str) -> str:
    """Resolve a secrets key_file against the secrets dir, warning when it escapes that dir.

    Returns the absolute path so a relative key_file is read against the secrets file's location
    (not the caller's cwd); without this, a non-default ``--secrets-path`` would make wallet-link
    read/check the wrong key file.
    """
    kf = Path(key_file).expanduser()
    kf_abs = os.path.abspath(kf if kf.is_absolute() else Path(base) / kf)
    try:
        outside = os.path.commonpath((base, kf_abs)) != base
    except ValueError:  # different drives / mixed path kinds
        outside = True
    if outside:
        # The threat model already grants mnemonic access; this just surfaces a
        # tampered worker_secrets.json that redirects a read outside the dir.
        # We don't reject because legitimate absolute key_file paths are normal.
        print(
            f"warning: key_file for {address} resolves outside the secrets dir "
            f"{base}; reading it anyway ({key_file})",
            file=sys.stderr,
        )
    return kf_abs


class SecretsLoadError(Exception):
    """Raised when a present worker-secrets file cannot be read or parsed.

    Distinct from an *absent* secrets file (which :func:`discover_keys` reports as no keys) so a
    caller can surface a corrupt/unreadable file instead of the misleading "no worker keys, create
    one" path.
    """


def discover_keys(secrets_path: str | Path) -> dict[str, _KeyEntry]:
    """Load WorkerManager secrets: {address: {"alias", "key_file"}}.

    Returns an empty mapping when the file is absent. Raises :class:`SecretsLoadError` when a
    present file is unreadable, not valid JSON, or not a JSON object, so a corrupt secrets file is
    not masked as "no keys".
    """
    path = Path(secrets_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        # Present-but-unreadable/corrupt is distinct from not-found: surface it instead of
        # masking it as the "no worker keys, create one" case.
        raise SecretsLoadError(f"could not read worker secrets at {secrets_path}: {exc}") from exc
    if not isinstance(raw, dict):
        # Valid JSON whose root is a list/scalar would crash on raw.items(); a malformed-but-
        # parseable secrets file is a corrupt file, not "no keys".
        raise SecretsLoadError(f"worker secrets at {secrets_path} is not a JSON object")
    base = os.path.dirname(os.path.abspath(path))
    # Keyed by address; warn (rather than silently overwrite) when two aliases share an address,
    # since the second would otherwise win invisibly — a footgun combined with relative key_files.
    keys: dict[str, _KeyEntry] = {}
    for alias, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        address = entry.get("address")
        key_file = entry.get("key_file")
        # Require strings (not just truthy): a tampered secrets file with non-string values
        # (e.g. {"address": 123}) would otherwise reach _checked_key_file and crash on
        # Path(123), defeating the documented "warn, skip" handling for a malformed file.
        if not (isinstance(address, str) and address and isinstance(key_file, str) and key_file):
            continue
        if address in keys:
            print(
                f"warning: duplicate address {address} in {secrets_path}; "
                f"alias {alias!r} overrides {keys[address]['alias']!r}",
                file=sys.stderr,
            )
        keys[address] = {
            "alias": alias,
            "key_file": _checked_key_file(base, address, key_file),
        }
    return keys


class _RequestError(SystemExit):
    """A wallet-link HTTP/network failure.

    Subclasses ``SystemExit`` so an unhandled error still aborts the CLI with a clean message
    (no traceback), like the rest of this module, while carrying the HTTP ``status`` (``None`` for
    network/transport errors) so the device-flow poll loop can stop on a terminal 4xx instead of
    retrying it until the deadline.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _is_terminal_poll_error(exc: BaseException) -> bool:
    """True when a poll error should stop the device flow immediately rather than be retried.

    Terminal = a 4xx client error other than 408/429 (e.g. the device code is expired, invalid, or
    already used): every retry returns the same response until the deadline. 5xx, 408, 429, network
    blips, and malformed-body errors (no ``status``) are transient and keep polling.
    """
    status = getattr(exc, "status", None)
    return status is not None and 400 <= status < 500 and status not in (408, 429)


def _loads_json_object(url: str, raw: str) -> dict[str, Any]:
    """Parse a server response body into a JSON object.

    Raises ``SystemExit`` when the body is not valid JSON, or is valid JSON that is not an object
    (a list, string, number, or null), so callers never hit an ``AttributeError`` from ``.get(...)``
    on a non-dict nor an unwrapped ``JSONDecodeError``. On the poll path this ``SystemExit`` is
    caught and retried as transient; on start/submit it aborts cleanly like the HTTP-error path.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"request to {url} returned non-JSON: {raw[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"request to {url} returned unexpected JSON type: {type(parsed).__name__}")
    return parsed


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects so a 3xx surfaces as an HTTPError.

    A signed wallet-link POST body must never be replayed to a redirect target: returning None
    from redirect_request makes urllib raise instead of re-issuing the request elsewhere. (The
    keep-alive ``_JsonPoster`` path uses ``http.client`` directly and never redirects either.)
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


# Opener without redirect following, used for the (non-keep-alive) start/submit POSTs.
_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _post_json(url: str, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    """POST JSON payload to url; raises SystemExit on HTTP/network errors or a non-object body."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return _loads_json_object(url, resp.read(_MAX_RESPONSE_BYTES).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # Filter the server-supplied error body so it can't inject terminal escapes (matches the
        # _printable() treatment of user_code / verification_uri_complete on the success path).
        detail = _printable(exc.read(_MAX_RESPONSE_BYTES).decode("utf-8", "replace"))
        raise _RequestError(f"request to {url} failed ({exc.code}): {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise _RequestError(f"could not reach {url}: {exc.reason}") from exc


def _printable(text: str) -> str:
    """Drop non-printable chars so server strings can't inject terminal escapes."""
    return "".join(c for c in text if c.isprintable())


def _default_port(scheme: str) -> int | None:
    """Return the default TCP port for an URL scheme (so an unspecified port compares equal)."""
    return {"https": 443, "http": 80}.get(scheme)


def _submit_rejection(submit: dict[str, Any]) -> str | None:
    """Return a user-facing message if /device/submit reported rejected signatures, else None.

    The server can return HTTP 200 while rejecting individual signatures (e.g.
    ``{"rejected": [{"address": "allo1...", "reason": "..."}]}``) or signalling a top-level
    ``error``. Surfacing it
    here lets the caller bail before the poll loop instead of waiting out the full deadline only to
    report ``Linked 0 worker(s)``. Server strings are filtered through ``_printable``.
    """
    rejected = submit.get("rejected")
    if isinstance(rejected, list) and rejected:
        lines = ["Server rejected one or more worker signatures:"]
        for item in rejected:
            if not isinstance(item, dict):
                continue
            addr = _printable(str(item.get("address", "?")))
            reason = _printable(str(item.get("reason", "no reason given")))
            lines.append(f"  - {addr}: {reason}")
        return "\n".join(lines)
    error = submit.get("error")
    if error:
        return f"Server rejected the signature submission: {_printable(str(error))}"
    return None


# @@TODO: This module hand-rolls an HTTP transport (_post_json + _JsonPoster: proxy resolution,
# CONNECT tunneling, Proxy-Authorization, keep-alive reconnect, bounded read) that duplicates the
# requests.Session transport allora-sdk-py's ForgeBackendClient already owns for the same Forge
# host. Consolidate by moving the device-flow transport into allora-sdk-py (e.g. a DeviceFlowClient
# reusing the SDK Session) so builder-kit keeps only the CLI orchestration + ADR-036 sign-doc
# builder. Cross-repo (allora-sdk-py + forge-v2); tracked as a follow-up, not done here.
class _JsonPoster:
    """Reusable JSON poster that holds one keep-alive connection to a fixed host.

    The device-flow poll loop hits a single Forge host up to ~120 times; ``urllib`` opens a
    fresh TCP+TLS connection per call, so reusing one connection removes a handshake per poll.
    This mirrors ``_post_json``'s bounded read, error sanitization, ``SystemExit``-on-failure, and
    proxy resolution (HTTP(S)_PROXY / NO_PROXY, including ``user:pass@`` proxy credentials sent as
    ``Proxy-Authorization``) so the poll loop's existing transient-error handling and proxied
    environments both keep working. A server that closed an idle keep-alive connection between polls
    is handled by one transparent reconnect.
    """

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        parts = urlsplit(base_url)
        self._host = parts.hostname or ""
        self._port = parts.port
        self._https = parts.scheme != "http"
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None
        # http.client does not read proxy env vars, so resolve the proxy the way urllib (used by
        # _post_json for /start and /submit) does. Without this the keep-alive poll connection
        # would silently go direct and hang/fail behind a corporate HTTP(S) proxy.
        proxy = self._select_proxy(base_url, self._https)
        self._proxy: tuple[str, int | None] | None = (proxy[0], proxy[1]) if proxy is not None else None
        # Pre-built "Basic <base64>" Proxy-Authorization value when the proxy URL carried userinfo,
        # else None — without it an authenticated proxy answers every poll with 407.
        self._proxy_auth: str | None = proxy[2] if proxy is not None else None

    @staticmethod
    def _select_proxy(base_url: str, https: bool) -> tuple[str, int | None, str | None] | None:
        """Resolve the proxy for base_url from the environment as ``(host, port, auth)``.

        Returns None for a direct connection, including when the host matches NO_PROXY. ``auth`` is
        a ready ``Proxy-Authorization`` header value (``Basic <base64>``) when the proxy URL carries
        userinfo, else None. Mirrors urllib's resolution (``getproxies`` + ``proxy_bypass``) so the
        poll path honors the same proxy configuration — credentials included — as the urllib-based
        start/submit requests.
        """
        host = urlsplit(base_url).hostname or ""
        if urllib.request.proxy_bypass(host):
            return None
        proxies = urllib.request.getproxies()
        proxy_url = proxies.get("https" if https else "http") or proxies.get("all")
        if not proxy_url:
            return None
        parsed = urlsplit(proxy_url if "://" in proxy_url else f"//{proxy_url}", scheme="http")
        if not parsed.hostname:
            return None
        auth = None
        if parsed.username is not None:
            # URL-unquote the userinfo before base64 so percent-encoded credentials (e.g. p%40ss)
            # decode to their literal bytes, matching how urllib builds Proxy-Authorization.
            user = unquote(parsed.username)
            password = unquote(parsed.password) if parsed.password is not None else ""
            token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
            auth = f"Basic {token}"
        if parsed.port is None:
            # http.client silently defaults a missing port to 443 (HTTPS) / 80 (HTTP); an operator
            # who set HTTPS_PROXY=proxy.corp.local expecting 3128/8080 would otherwise hit a
            # confusing connection failure. Surface the implicit default.
            print(
                f"warning: proxy {parsed.hostname} has no explicit port; defaulting to "
                f"{443 if https else 80}",
                file=sys.stderr,
            )
        return (parsed.hostname, parsed.port, auth)

    def _connect(self) -> http.client.HTTPConnection:
        if self._proxy is not None:
            proxy_host, proxy_port = self._proxy
            if self._https:
                # CONNECT-tunnel the TLS session through the proxy so the certificate is still
                # validated against the real Forge host rather than the proxy. Proxy credentials
                # (if any) ride on the CONNECT request itself via set_tunnel's headers.
                conn = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=self._timeout)
                tunnel_headers = {"Proxy-Authorization": self._proxy_auth} if self._proxy_auth else {}
                conn.set_tunnel(self._host, self._port, headers=tunnel_headers)
                return conn
            return http.client.HTTPConnection(proxy_host, proxy_port, timeout=self._timeout)
        if self._https:
            return http.client.HTTPSConnection(self._host, self._port, timeout=self._timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to url over the held connection; raises SystemExit on HTTP/network errors."""
        # Through a plain-HTTP proxy the request line must carry the absolute URL (RFC 7230 5.3.2);
        # direct and HTTPS-tunneled connections use the origin-form path.
        if self._proxy is not None and not self._https:
            request_target = url
        else:
            request_target = urlsplit(url).path or "/"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        # Plain-HTTP proxy: auth rides the request; HTTPS auth already rode the CONNECT tunnel.
        if self._proxy is not None and not self._https and self._proxy_auth:
            headers["Proxy-Authorization"] = self._proxy_auth
        # Retry once: the server may have dropped an idle keep-alive connection between polls.
        for attempt in (1, 2):
            if self._conn is None:
                self._conn = self._connect()
            try:
                self._conn.request("POST", request_target, body=body, headers=headers)
                resp = self._conn.getresponse()
                data = resp.read(_MAX_RESPONSE_BYTES)
                if resp.status >= 400:
                    detail = _printable(data.decode("utf-8", "replace"))
                    # Close before raising: _RequestError (a BaseException) escapes the except below,
                    # so an undrained connection would defeat keep-alive.
                    self.close()
                    raise _RequestError(f"request to {url} failed ({resp.status}): {detail}", status=resp.status)
                return _loads_json_object(url, data.decode("utf-8", "replace"))
            except (http.client.HTTPException, OSError) as exc:
                self.close()
                if attempt == 2:
                    raise _RequestError(f"could not reach {url}: {exc}") from exc
        raise _RequestError(f"could not reach {url}")  # unreachable: the loop returns or raises

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


def run_link(
    forge_url: str = DEFAULT_FORGE_URL,
    secrets_path: str = DEFAULT_SECRETS_PATH,
    addresses: list[str] | None = None,
    open_browser: bool = True,
    insecure: bool = False,
) -> int:
    """Drive the full device flow. Returns a process exit code."""
    forge_url = forge_url.rstrip("/")
    parsed = urlparse(forge_url)
    if (
        parsed.scheme != "https"
        and parsed.hostname not in _LOOPBACK_HOSTS
        and not insecure
    ):
        print(
            f"refusing plaintext forge URL {forge_url} "
            f"(use --insecure to override for local dev)",
            file=sys.stderr,
        )
        return 1
    if parsed.path and parsed.path != "/":
        # The /api/v1/wallet-link/... prefix is appended below, so a forge-url carrying a path
        # (e.g. https://forge.allora.network/api/v1) would produce a double /api/v1 and a confusing
        # 404. Reject it up front with an actionable message instead.
        print(
            f"forge_url must be a scheme+host with no path; got path={parsed.path!r}. "
            "Use e.g. https://forge.allora.network (the /api/v1 prefix is added automatically).",
            file=sys.stderr,
        )
        return 1
    try:
        keys = discover_keys(secrets_path)
    except SecretsLoadError as exc:
        # A corrupt / unreadable secrets file is a distinct failure from "no keys yet"; report it
        # instead of the misleading "create a worker first" hint.
        print(str(exc), file=sys.stderr)
        return 1
    if not keys:
        print(
            f"No worker keys found in {secrets_path}. Create a worker first "
            f"(e.g. via WorkerManager.deploy_worker).",
            file=sys.stderr,
        )
        return 1

    selected = addresses or list(keys.keys())
    missing = [a for a in selected if a not in keys]
    if missing:
        # A managed-custody worker has no local key file, so it legitimately won't appear
        # here. Spell that out rather than leave the operator thinking a valid worker is
        # broken: managed workers are linked automatically via the backend, not via this CLI.
        print(
            f"No local key for: {', '.join(missing)}\n"
            "  (this command links LOCAL-custody workers only; a managed-custody worker is "
            "linked automatically by the Forge backend and needs no local signing)",
            file=sys.stderr,
        )
        return 1

    # Validate key files up front so a stale secrets entry fails before we
    # open a server-side device session that would otherwise be orphaned.
    unreadable = [a for a in selected if not Path(keys[a]["key_file"]).is_file()]
    if unreadable:
        print(
            f"key file missing for: {', '.join(unreadable)} (stale {secrets_path}?)",
            file=sys.stderr,
        )
        return 1

    # Pre-validate the optional cosmpy dependency here, before /device/start opens a server-side
    # session. sign_challenge imports cosmpy lazily; without this check a missing wallet-link extra
    # would fail only after the session exists, orphaning it until the janitor reaps it (~15 min).
    try:
        from cosmpy.aerial.wallet import LocalWallet  # noqa: F401
    except ImportError:
        print(
            "cosmpy is required to sign. Install the wallet-link extra "
            "(pip install 'allora-forge-builder-kit[wallet-link]') or cosmpy==0.11.1.",
            file=sys.stderr,
        )
        return 1

    print(f"Linking {len(selected)} worker address(es) to Allora Forge at {forge_url}")

    # 1. Start the device session.
    start = _post_json(
        f"{forge_url}/api/v1/wallet-link/device/start", {"addresses": selected}
    )
    for field in ("device_code", "user_code", "verification_uri_complete"):
        value = start.get(field)
        # Require a non-empty string, not just truthiness: a non-string (e.g. an int from an
        # over-eager JSON marshaler) would later crash urlparse()/_printable() with a raw traceback.
        if not value or not isinstance(value, str):
            print(f"server response missing or malformed required field: {field}", file=sys.stderr)
            return 1
    device_code = start["device_code"]
    user_code = start["user_code"]
    verification_uri_complete = start["verification_uri_complete"]
    try:
        interval = int(start.get("interval", 5))
    except (TypeError, ValueError):
        interval = 5
    interval = max(1, min(interval, 60))
    challenges = {
        c["address"]: c["message"]
        for c in start.get("challenges", [])
        if isinstance(c, dict) and c.get("address") and c.get("message")
    }

    # Pin the server-returned approval URL to the forge origin and a safe
    # scheme so a compromised server can't phish via a different host or hand
    # a file://, javascript:, or app-launcher URI to the OS handler.
    verification = urlparse(verification_uri_complete)
    same_host = verification.hostname == parsed.hostname
    # The port is part of the origin: pin it too (using the scheme default when unspecified) so a
    # compromised server can't redirect to an arbitrary port on the same host.
    same_port = (verification.port or _default_port(verification.scheme)) == (
        parsed.port or _default_port(parsed.scheme)
    )
    safe_scheme = verification.scheme == "https" or (
        verification.scheme == "http"
        and (verification.hostname in _LOOPBACK_HOSTS or insecure)
    )
    if not (same_host and same_port and safe_scheme):
        print(
            f"refusing to open untrusted verification URL: {verification_uri_complete}",
            file=sys.stderr,
        )
        return 1

    # 2. Sign each challenge locally and submit the signatures.
    signatures = []
    for address in selected:
        message = challenges.get(address)
        if message is None:
            print(f"Server returned no challenge for {address}", file=sys.stderr)
            return 1
        try:
            # utf-8 (not the platform default) so a BOM/exotic-locale key file fails with a clear
            # UnicodeDecodeError here rather than a confusing downstream error.
            mnemonic = Path(keys[address]["key_file"]).read_text(encoding="utf-8").strip()
        except OSError as exc:
            # OSError messages describe the file (path/permissions), never key material.
            print(f"failed to read key file for {address}: {exc}", file=sys.stderr)
            return 1
        if not mnemonic:
            # Catch this before cosmpy, which would otherwise raise a confusing internal error.
            print(f"key file for {address} is empty: {keys[address]['key_file']}", file=sys.stderr)
            return 1
        try:
            pubkey_b64, signature_b64 = sign_challenge(mnemonic, address, message)
        except WalletSignError as exc:
            # Raised by sign_challenge only with a key-material-free message (e.g. address mismatch).
            print(f"failed to sign challenge for {address}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            # Any other signing/derivation failure (cosmpy ValueError, Bip39/secp256k1 errors, a bad
            # UTF-8 mnemonic, ...) can embed mnemonic fragments in its message or args, so surface
            # only the exception *type* — never str(exc) — keeping key material out of stderr/logs.
            print(
                f"failed to sign challenge for {address}: {type(exc).__name__} "
                "(detail withheld to avoid leaking key material; ensure the key file holds a "
                "valid BIP-39 mnemonic for this address)",
                file=sys.stderr,
            )
            return 1
        signatures.append(
            {"address": address, "pubkey": pubkey_b64, "signature": signature_b64}
        )

    submit = _post_json(
        f"{forge_url}/api/v1/wallet-link/device/submit",
        {"device_code": device_code, "signatures": signatures},
    )
    rejection = _submit_rejection(submit)
    if rejection is not None:
        print(rejection, file=sys.stderr)
        return 1

    # 3. Hand off to the browser for the logged-in user to approve.
    print()
    print(f"  First copy your one-time code: {_printable(user_code)}")
    print(f"  Then approve the link at: {_printable(verification_uri_complete)}")
    print()
    if open_browser:
        try:
            opened = webbrowser.open(verification_uri_complete)
        except Exception:  # pragma: no cover - some platforms raise instead of returning False
            opened = False
        if opened:
            print("Opened your browser. Waiting for approval...")
        else:
            print("Could not open a browser; open the URL above. Waiting...")
    else:
        print("Waiting for approval...")

    # 4. Poll until the user approves/denies or the session expires. Honor the
    # server-advertised expires_in (clamped) so the client deadline tracks the
    # real session TTL instead of a fixed local constant.
    try:
        timeout = int(start.get("expires_in", _POLL_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = _POLL_TIMEOUT_SECONDS
    timeout = max(1, min(timeout, _POLL_TIMEOUT_SECONDS))
    # Monotonic deadline: immune to NTP steps / manual clock changes / DST that a wall-clock
    # time.time() deadline would let silently extend or prematurely abort the session.
    deadline = time.monotonic() + timeout
    # Reuse one keep-alive connection across the (up to ~120) polls to the same Forge host
    # instead of a fresh TCP+TLS handshake per poll.
    poller = _JsonPoster(forge_url)
    try:
        while time.monotonic() < deadline:
            time.sleep(max(1, interval))
            try:
                poll = poller.post(
                    f"{forge_url}/api/v1/wallet-link/device/poll", {"device_code": device_code}
                )
            except SystemExit as exc:
                if _is_terminal_poll_error(exc):
                    # Stop now with the server's explanation instead of spamming "retrying..."
                    # until the deadline — a terminal 4xx returns the same response every poll.
                    print(f"\nLink failed: {exc}", file=sys.stderr)
                    return 1
                # Transient (5xx / 408 / 429 / DNS blip / malformed body): keep polling until our
                # monotonic deadline instead of aborting the whole flow.
                print(f"  (poll error: {exc}; retrying...)", file=sys.stderr)
                continue
            status = poll.get("status")
            if status == "approved":
                # Coerce defensively: a server emitting {"linked": null} would make
                # poll.get("linked", []) return None (the key exists) and crash len(None).
                linked_raw = poll.get("linked")
                linked = linked_raw if isinstance(linked_raw, list) else []
                # Server-controlled strings: filter terminal escapes so a malicious 'linked' entry
                # can't render a clickable OSC-8 hyperlink disguised as a bech32 address.
                linked_addrs = {a for a in linked if isinstance(a, str)}
                missing = [a for a in selected if a not in linked_addrs]
                if missing:
                    # Approved but the backend linked only a subset (or none): a partial link is a
                    # failure, not a silent exit 0 that signals success to a CI pipeline.
                    print(
                        f"\nLink reported approved but {len(missing)} requested address(es) "
                        "were not linked:\n  " + "\n  ".join(_printable(a) for a in missing),
                        file=sys.stderr,
                    )
                    return 1
                print(f"\nLinked {len(linked_addrs)} verified worker(s):")
                for addr in sorted(linked_addrs):
                    print(f"  + {_printable(addr)}")
                return 0
            if status == "denied":
                print("\nLink request was denied in the browser.", file=sys.stderr)
                return 1
            if status == "expired":
                print("\nLink request expired before approval.", file=sys.stderr)
                return 1
    finally:
        poller.close()

    print("\nTimed out waiting for browser approval.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="allora-forge-link",
        description=(
            "Prove ownership of LOCAL-custody worker wallets and link them to Allora Forge. "
            "This command applies only to workers whose signing key lives in a local key file "
            "(listed in the WorkerManager secrets file). Managed-custody workers are linked "
            "automatically by the Forge backend, hold no local key to prove, and are therefore "
            "neither required nor handled here."
        ),
    )
    parser.add_argument("--forge-url", default=DEFAULT_FORGE_URL, help="Forge base URL")
    parser.add_argument(
        "--secrets-path", default=DEFAULT_SECRETS_PATH, help="WorkerManager secrets file"
    )
    parser.add_argument(
        "--address",
        action="append",
        dest="addresses",
        help="Limit to specific allo1... address(es); repeatable. Default: all local keys.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not auto-open a browser"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow a plaintext http:// forge URL (local dev only)",
    )
    args = parser.parse_args(argv)
    return run_link(
        forge_url=args.forge_url,
        secrets_path=args.secrets_path,
        addresses=args.addresses,
        open_browser=not args.no_browser,
        insecure=args.insecure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
