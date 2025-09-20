#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command string to run in each subdir>"
  exit 2
fi

cmd="$1"

while IFS= read -r -d '' d; do
  if [[ -f "$d/pyproject.toml" ]]; then
    echo "———"
    echo "▶ Running in: $d"
    pushd "$d" >/dev/null
    bash -o pipefail -c "$cmd"
    popd >/dev/null
  else
    echo "⏭ Skipping $d (no pyproject.toml)"
  fi
done < <(find . -mindepth 1 -maxdepth 1 -type d -print0)
