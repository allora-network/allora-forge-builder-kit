import os
from getpass import getpass

def get_api_key(api_key_file=".allora_api_key"):
    """
    Load API key from file if available, otherwise prompt and save it.
    """
    if os.path.exists(api_key_file):
        with open(api_key_file, "r") as f:
            return f.read().strip()

    key = getpass("Enter your Allora API key: ").strip()
    with open(api_key_file, "w") as f:
        f.write(key)
    return key