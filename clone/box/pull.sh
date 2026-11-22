#!/bin/bash
# From rsi-brain: copy /vol/runs back from a box.
# Usage: box/pull.sh "ssh -p PORT root@HOST" RUN_SUBDIR [DEST]
set -euo pipefail
SSH="$1"; SUB="${2:-.}"; DST="${3:-./results/pulled}"
mkdir -p "$DST"
$SSH "tar -C /vol/runs -czf - $SUB" | tar -C "$DST" -xzf -
echo "pulled /vol/runs/$SUB -> $DST"
