#!/bin/bash
# Build the incumbent and deliver requested artifacts as the oracle baseline.
# Pinned flags and deterministic decks must match the verifier reference.

set -euo pipefail

RANKS="${RANKS:-4}"
FLAGS="-O3 -fdefault-real-8 -ffp-contract=off -fallow-argument-mismatch -std=legacy -I/usr/include -L/usr/lib -lfftw3"

make -C /app/LAPS/src_compressible/2D fftwpath=/usr OPTIONS="$FLAGS"
test -x /app/LAPS/src_compressible/2D/mhd.exe
make -C /app/LAPS/src_compressible fftwpath=/usr OPTIONS="$FLAGS"
test -x /app/LAPS/src_compressible/mhd.exe

: > /tmp/walltimes.txt
for c in aw-2d-256 aw-2d-512 aw-128; do
  case "$c" in
    aw-2d-*) EXE=/app/LAPS/src_compressible/2D/mhd.exe ;;
    *)       EXE=/app/LAPS/src_compressible/mhd.exe ;;
  esac
  mkdir -p "/app/run/$c" "/logs/artifacts/$c"
  cd "/app/run/$c"
  cp "/app/decks/$c/mhd.input" ./mhd.input
  t0=$(date +%s%N)
  mpirun --allow-run-as-root --oversubscribe -np "$RANKS" "$EXE"
  t1=$(date +%s%N)
  echo "$c $t0 $t1" >> /tmp/walltimes.txt
  cp out*.dat times.dat "/logs/artifacts/$c/"
done

python3 - <<'EOF'
import json
rows = {}
for line in open('/tmp/walltimes.txt'):
    c, t0, t1 = line.split()
    rows[c] = round((int(t1) - int(t0)) / 1e9, 3)
with open('/logs/artifacts/timing.json', 'w') as f:
    json.dump(rows, f, indent=1)
EOF

echo "oracle: delivered"
ls -la /logs/artifacts/ /logs/artifacts/*/ | head -40
