"""Blacklist configuration for demo unit tests.

This module defines demos that should be excluded from deterministic replay tests. Each
entry includes a pattern to match against demo paths and a reason for exclusion.
"""

from pathlib import Path

# Blacklist for deterministic demo replay tests
# Format: {pattern: reason}
# Pattern can be any substring that appears in the demo path
DETERMINISTIC_REPLAY_BLACKLIST = {
    "DynScoopPour": (
        "Non-deterministic behavior in physics simulation. "
        "Test passes on local machines but fails inconsistently on GitHub Actions CI."
    ),
}


def is_demo_blacklisted(
    demo_path: Path, blacklist: dict[str, str]
) -> tuple[bool, str | None]:
    """Check if a demo path matches any blacklist pattern.

    Args:
        demo_path: Path to the demo file (as string or Path object)
        blacklist: Dictionary mapping patterns to reasons for blacklisting

    Returns:
        Tuple of (is_blacklisted, reason)
        - is_blacklisted: True if demo matches any blacklist pattern
        - reason: The reason string if blacklisted, None otherwise
    """
    demo_path_str = str(demo_path)
    for pattern, reason in blacklist.items():
        if pattern in demo_path_str:
            return True, reason
    return False, None
