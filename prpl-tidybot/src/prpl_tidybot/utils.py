"""Utility functions for the real tidybot."""

import atexit
import os
from pathlib import Path


def create_pid_file(name):
    """Utility to create a PID file."""
    # Check if PID file already exists
    pid_file_path = Path(f"/tmp/{name}.pid")
    if pid_file_path.exists():
        # Get PID of other process from lock file
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())

        # Check if PID matches current process
        if pid != os.getpid():
            # PID does not match current process, check if other process is still running
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale PID file (PID {pid})")
                pid_file_path.unlink()
            else:
                raise Exception(  # pylint: disable=broad-exception-raised
                    f"Another instance of the {name} is already running (PID {pid})"
                )

    # Write PID of current process to the file
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file_path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")

    # Register cleanup function to remove PID file upon exit
    atexit.register(remove_pid_file, pid_file_path)


def remove_pid_file(pid_file_path):
    """Utility to remove a PID file."""
    # Remove PID file if it corresponds to the current process
    if pid_file_path.exists():
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if pid == os.getpid():
            pid_file_path.unlink()
