#!/bin/bash
# Ship clone/ and xkv's src/ to a box (or clone the public repo on the box instead).
# Usage: box/push.sh "ssh -p PORT root@HOST"   (results come back with box/pull.sh)
set -euo pipefail
SSH="$1"; HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
tar -C "$HERE/.." -czf - --exclude=.git --exclude=__pycache__ "$(basename "$HERE")" | $SSH "mkdir -p /work && tar -C /work -xzf - && rm -rf /work/clone && mv /work/$(basename "$HERE") /work/clone"
tar -C "$ROOT" -czf - --exclude=__pycache__ src | $SSH "mkdir -p /work/xkv && tar -C /work/xkv -xzf -"
echo "pushed to $SSH:/work/clone and /work/xkv/src"
