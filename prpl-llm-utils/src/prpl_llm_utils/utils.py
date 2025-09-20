"""Utility functions."""

import hashlib
from typing import Any


def consistent_hash(obj: Any) -> int:
    """A hash function that is consistent between sessions, unlike hash()."""
    obj_str = repr(obj)
    obj_bytes = obj_str.encode("utf-8")
    hash_hex = hashlib.sha256(obj_bytes).hexdigest()
    hash_int = int(hash_hex, 16)
    # Mimic Python's built-in hash() behavior by returning a 64-bit signed int.
    # This makes it comparable to hash()'s output range.
    return hash_int if hash_int < 2**63 else hash_int - 2**6
