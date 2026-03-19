import os
from getpass import getpass

_API_KEY_SEARCH_PATHS = (
    ".allora_api_key",
    "notebooks/.allora_api_key",
)


def get_api_key(api_key_file: str | None = None) -> str:
    """Resolve the Allora API key.

    Resolution order:
      1. ``ALLORA_API_KEY`` environment variable
      2. Explicit *api_key_file* (if provided)
      3. Well-known file paths (``.allora_api_key``, ``notebooks/.allora_api_key``)
      4. Interactive prompt (saves to *api_key_file* or ``.allora_api_key``)

    Get a free key at https://developer.allora.network
    """
    env = os.environ.get("ALLORA_API_KEY", "").strip()
    if env:
        return env

    search = [api_key_file] if api_key_file else list(_API_KEY_SEARCH_PATHS)
    for path in search:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                val = f.read().strip()
            if val:
                return val

    print("No Allora API key found.")
    print("Get a free key at: https://developer.allora.network")
    print("Then set ALLORA_API_KEY env var, or enter it below.\n")
    key = getpass("Enter your Allora API key: ").strip()
    if not key:
        raise RuntimeError(
            "Allora API key is required. "
            "Sign up at https://developer.allora.network"
        )
    dest = api_key_file or ".allora_api_key"
    fd = os.open(dest, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key.encode())
    finally:
        os.close(fd)
    return key