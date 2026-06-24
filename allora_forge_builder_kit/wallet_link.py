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
import json
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

DEFAULT_FORGE_URL = "https://forge.allora.network"
DEFAULT_SECRETS_PATH = "worker_secrets.json"
_POLL_TIMEOUT_SECONDS = 600


def build_adr036_sign_doc(signer: str, message: str) -> bytes:
    """Return the canonical Cosmos ADR-036 amino StdSignDoc bytes.

    Must match the Go verifier and Keplr byte-for-byte: keys sorted alphabetically
    at every level, no whitespace, ``data`` = standard-base64 of the raw message.
    """
    data = base64.standard_b64encode(message.encode("utf-8")).decode("ascii")
    return (
        '{"account_number":"0","chain_id":"","fee":{"amount":[],"gas":"0"},'
        '"memo":"","msgs":[{"type":"sign/MsgSignData","value":{"data":"'
        + data
        + '","signer":"'
        + signer
        + '"}}],"sequence":"0"}'
    ).encode("utf-8")


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
            "cosmpy is required to sign. Install the builder kit deps "
            "(pip install -e . / cosmpy==0.11.1)."
        ) from exc

    wallet = LocalWallet.from_mnemonic(mnemonic, "allo")
    derived = str(wallet.address())
    if derived != address:
        raise ValueError(
            f"key derives address {derived}, which does not match requested {address}"
        )

    signer = wallet.signer()
    doc = build_adr036_sign_doc(address, message)
    signature = signer.sign(doc)
    pubkey_b64 = base64.standard_b64encode(signer.public_key_bytes).decode("ascii")
    signature_b64 = base64.standard_b64encode(signature).decode("ascii")
    return pubkey_b64, signature_b64


def discover_keys(secrets_path: str | Path) -> dict[str, dict]:
    """Load WorkerManager secrets: {address: {"alias", "key_file"}}."""
    path = Path(secrets_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, dict] = {}
    for alias, entry in raw.items():
        if isinstance(entry, dict) and entry.get("address") and entry.get("key_file"):
            out[entry["address"]] = {"alias": alias, "key_file": entry["key_file"]}
    return out


def _post_json(url: str, payload: dict, timeout: float = 15.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"request to {url} failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach {url}: {exc.reason}") from exc


def run_link(
    forge_url: str = DEFAULT_FORGE_URL,
    secrets_path: str = DEFAULT_SECRETS_PATH,
    addresses: Optional[list[str]] = None,
    open_browser: bool = True,
) -> int:
    """Drive the full device flow. Returns a process exit code."""
    forge_url = forge_url.rstrip("/")
    keys = discover_keys(secrets_path)
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
        print(f"No local key for: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"Linking {len(selected)} worker address(es) to Allora Forge at {forge_url}")

    # 1. Start the device session.
    start = _post_json(
        f"{forge_url}/api/v1/wallet-link/device/start", {"addresses": selected}
    )
    device_code = start["device_code"]
    user_code = start["user_code"]
    verification_uri_complete = start["verification_uri_complete"]
    try:
        interval = int(start.get("interval", 5))
    except (TypeError, ValueError):
        interval = 5
    interval = max(1, min(interval, 60))
    challenges = {c["address"]: c["message"] for c in start.get("challenges", [])}

    # 2. Sign each challenge locally and submit the signatures.
    signatures = []
    for address in selected:
        message = challenges.get(address)
        if message is None:
            print(f"Server returned no challenge for {address}", file=sys.stderr)
            return 1
        mnemonic = Path(keys[address]["key_file"]).read_text().strip()
        pubkey_b64, signature_b64 = sign_challenge(mnemonic, address, message)
        signatures.append(
            {"address": address, "pubkey": pubkey_b64, "signature": signature_b64}
        )

    _post_json(
        f"{forge_url}/api/v1/wallet-link/device/submit",
        {"device_code": device_code, "signatures": signatures},
    )

    # 3. Hand off to the browser for the logged-in user to approve.
    print()
    print(f"  First copy your one-time code: {user_code}")
    print(f"  Then approve the link at: {verification_uri_complete}")
    print()
    if open_browser:
        try:
            webbrowser.open(verification_uri_complete)
            print("Opened your browser. Waiting for approval...")
        except Exception:  # pragma: no cover - headless / no browser
            print("Could not open a browser; open the URL above. Waiting...")
    else:
        print("Waiting for approval...")

    # 4. Poll until the user approves/denies or the session expires.
    deadline = time.time() + _POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(max(1, interval))
        try:
            poll = _post_json(
                f"{forge_url}/api/v1/wallet-link/device/poll", {"device_code": device_code}
            )
        except SystemExit as exc:
            # Transient HTTP/network error (502/503/429, DNS blip): keep polling
            # until our wall-clock deadline instead of aborting the whole flow.
            print(f"  (poll error: {exc}; retrying...)", file=sys.stderr)
            continue
        status = poll.get("status")
        if status == "approved":
            linked = poll.get("linked", [])
            print(f"\nLinked {len(linked)} verified worker(s):")
            for addr in linked:
                print(f"  + {addr}")
            return 0
        if status == "denied":
            print("\nLink request was denied in the browser.", file=sys.stderr)
            return 1
        if status == "expired":
            print("\nLink request expired before approval.", file=sys.stderr)
            return 1

    print("\nTimed out waiting for browser approval.", file=sys.stderr)
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="allora-forge-link",
        description="Prove ownership of local worker wallets and link them to Allora Forge.",
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
    args = parser.parse_args(argv)
    return run_link(
        forge_url=args.forge_url,
        secrets_path=args.secrets_path,
        addresses=args.addresses,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
