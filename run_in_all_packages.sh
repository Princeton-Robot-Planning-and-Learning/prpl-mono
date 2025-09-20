#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command string to run in each subdir>"
  exit 2
fi

cmd="$1"
status=0

while IFS= read -r -d '' d; do
  if [[ -f "$d/pyproject.toml" ]]; then
    echo "———"
    echo "▶ Running in: $d"
    pushd "$d" >/dev/null
    if ! bash -o pipefail -c "$cmd"; then
      echo "❌ Failed in $d"
      status=1
    else
      echo "✅ Passed in $d"
    fi
    popd >/dev/null
  else
    echo "⏭ Skipping $d (no pyproject.toml)"
  fi
done < <(find . -mindepth 1 -maxdepth 1 -type d -print0)

exit "$status"
