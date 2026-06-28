#!/usr/bin/env bash
# HiveMind launcher — checks deps, then runs the cluster (or test suite).
# Usage:
#   ./hivemind              # prompt for task
#   ./hivemind "my task"    # run single task
#   ./hivemind -i           # interactive mode
#   ./hivemind --test       # run all provider tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GRN="\033[32m"; RED="\033[31m"; YLW="\033[33m"; DIM="\033[2m"; RST="\033[0m"
ok()   { echo -e "  ${GRN}✓${RST} $*"; }
err()  { echo -e "  ${RED}✗${RST} $*" >&2; }
info() { echo -e "  ${DIM}$*${RST}"; }

echo ""
echo -e "${YLW}⬡ HiveMind${RST}"

# ── 1. Python version check ───────────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=${version%%.*}
        minor=${version##*.}
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.10+ required — not found on PATH"
    info "Install: https://python.org/downloads"
    info "Current python3 version: $(python3 --version 2>/dev/null || echo 'not found')"
    exit 1
fi
ok "Python $($PYTHON --version | awk '{print $2}')"

# ── 2. .env check ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    err ".env not found"
    echo ""
    echo -e "  Quick setup:"
    echo -e "    cp .env.example .env"
    echo -e "    nano .env           # add at least one provider key"
    echo -e "    ./hivemind --test   # verify it works"
    echo ""
    exit 1
fi
ok ".env found"

# ── 3. Dependency check ───────────────────────────────────────────────────────
MISSING=()
while IFS= read -r line || [ -n "$line" ]; do
    # Strip version specifiers and comments
    pkg=$(echo "$line" | sed 's/[>=<!].*//' | sed 's/#.*//' | tr -d '[:space:]')
    [ -z "$pkg" ] && continue
    # pip show uses the package name; map common import-name differences
    if ! "$PYTHON" -m pip show "$pkg" &>/dev/null 2>&1; then
        MISSING+=("$pkg")
    fi
done < requirements.txt

if [ ${#MISSING[@]} -gt 0 ]; then
    info "Installing missing packages: ${MISSING[*]}"
    "$PYTHON" -m pip install -q -r requirements.txt
    ok "Dependencies installed"
else
    ok "Dependencies satisfied"
fi

echo ""

# ── 4. Dispatch ───────────────────────────────────────────────────────────────
if [ "${1:-}" = "--test" ]; then
    shift
    exec "$PYTHON" tests/test_all.py "$@"
else
    exec "$PYTHON" main.py "$@"
fi
