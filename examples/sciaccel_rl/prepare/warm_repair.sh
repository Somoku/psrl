#!/usr/bin/env bash
# warm_repair.sh — re-warm the task images that failed a nop warm pass.
#
# A warm pass on a COLD node loses a few tasks to transient registry errors: the build
# fails with a bare `EOF` on the metadata HEAD for a base image (observed on
# `python:3.13-slim`, the verifier base), even though the same request succeeds 10/10
# sequentially from that host. Those tasks are not broken, they just never finished
# building, so the first training step that samples one pays a full cold build
# (~1-3 min) instead of ~16 s.
#
# The trigger is a MISSING BASE IMAGE, not concurrency and not a cold task cache. The
# mirror resolves to a link-local address (169.254.0.51), i.e. a node-local proxy, and it
# intermittently drops metadata requests -- both `EOF` and `dial tcp ...: i/o timeout` --
# while sequential curls to the same URL return 200. A node with env_setup already down
# to 15 s still lost 13 of 129 tasks this way, because the verifier's base image is a
# separate FROM that env-image layer caching never supplies.
#
# provision_docker_nodes.sh now pre-pulls python:3.13-slim, debian:bookworm-slim and
# alpine:3.19, which removes the request entirely. Run it before a warm pass and this
# script should find nothing to repair. Concurrency is at most a contributing factor and
# is NOT sufficient on its own: two nodes running the identical 12-way pass lost 20 tasks
# and 0 tasks respectively.
#
# This reads a completed pass's results.jsonl, picks out the tasks whose error_class is
# not `ok`, and re-runs ONLY those at low concurrency. Idempotent: with nothing to
# repair it exits 0 without launching anything.
#
# Usage:
#   bash warm_repair.sh --results <nop_warm_dir>/results.jsonl
#   bash warm_repair.sh --results <dir>/results.jsonl --host 28.49.195.154
#   bash warm_repair.sh --results <dir>/results.jsonl --dry-run
#
# Options:
#   --results PATH    results.jsonl from the warm pass to repair. (required)
#   --host HOST       Run the repair on this host over ssh. Default: locally.
#   --concurrency N   Tasks in flight (default: 4). Low on purpose: these builds are
#                     the ones that already failed once, so the repair trades speed for
#                     the best chance of completing.
#   --dataset PATH    v2 parquet (default: examples/sciaccel_rl/data/v2/all.parquet).
#   --dry-run         List the failed tasks and the command, run nothing.
#   -h | --help       Print this help.

set -euo pipefail

PSRL_PATH=${PSRL_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}

RESULTS=""
HOST=""
CONCURRENCY=4
DATASET="${PSRL_PATH}/examples/sciaccel_rl/data/v2/all.parquet"
DRY_RUN=0

usage() { sed -n '2,41p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results)      RESULTS="$2"; shift 2 ;;
        --host)         HOST="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${RESULTS}" ]] || { echo "ERROR: --results is required." >&2; usage >&2; exit 2; }
[[ -f "${RESULTS}" ]] || { echo "ERROR: results file not found: ${RESULTS}" >&2; exit 2; }

# Task names are read with python rather than grep/jq: the exception text in these
# records contains newlines and quotes, so line-oriented parsing mangles it.
mapfile -t FAILED < <(python3 -c "
import json, sys
seen = []
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if (r.get('error_class') or 'ok') != 'ok':
        name = r.get('task_name', '')
        if name and name not in seen:
            seen.append(name)
print('\n'.join(seen))
" "${RESULTS}")

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "Nothing to repair: every task in ${RESULTS} has error_class=ok."
    exit 0
fi

echo "=== warm_repair ==="
echo "  results     : ${RESULTS}"
echo "  host        : ${HOST:-<local>}"
echo "  concurrency : ${CONCURRENCY}"
echo "  failed tasks: ${#FAILED[@]}"
for t in "${FAILED[@]}"; do echo "    ${t}"; done
echo

OUT_DIR="$(dirname "${RESULTS}")_repair_$(date +%m%d_%H%M%S)"

# One eval invocation per task. --task-glob takes a single fnmatch pattern, so a
# combined run is not possible, and separate runs also mean one hard failure does not
# abort the rest.
run_one() {
    local task="$1" out="$2"
    cd "${PSRL_PATH}"
    bash examples/sciaccel_rl/eval/run_eval.sh \
        --agent nop \
        --dataset "${DATASET}" \
        --task-glob "${task}" \
        --output-dir "${out}" \
        --skip-gpu-tasks \
        --max-per-instance "${CONCURRENCY}" \
        -n "${CONCURRENCY}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run) would run, one per task:"
    echo "  bash examples/sciaccel_rl/eval/run_eval.sh --agent nop \\"
    echo "      --dataset ${DATASET} --task-glob <task> \\"
    echo "      --output-dir ${OUT_DIR}/<n> --skip-gpu-tasks \\"
    echo "      --max-per-instance ${CONCURRENCY} -n ${CONCURRENCY}"
    exit 0
fi

n_ok=0
n_fail=0
# Create the log directory once, up front. Both branches below redirect to
# "${out}.log", and a missing parent makes the redirect itself fail -- which bash
# reports as the command failing, so all 13 tasks looked like build failures when
# nothing had even been attempted.
mkdir -p "${OUT_DIR}"
for i in "${!FAILED[@]}"; do
    task="${FAILED[$i]}"
    out="${OUT_DIR}/$(printf '%02d' "$i")"
    echo "--- repairing ${task}"
    if [[ -n "${HOST}" ]]; then
        # Quoted once for ssh, which concatenates its arguments and lets the REMOTE
        # shell re-split them.
        cmd="cd ${PSRL_PATH} && bash examples/sciaccel_rl/eval/run_eval.sh --agent nop \
--dataset ${DATASET} --task-glob $(printf '%q' "${task}") --output-dir ${out} \
--skip-gpu-tasks --max-per-instance ${CONCURRENCY} -n ${CONCURRENCY}"
        if ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${HOST}" \
               bash -lc "$(printf '%q' "${cmd}")" >"${out}.log" 2>&1; then
            n_ok=$((n_ok + 1))
        else
            n_fail=$((n_fail + 1))
            echo "    FAILED, see ${out}.log"
        fi
    else
        mkdir -p "$(dirname "${out}")"
        if run_one "${task}" "${out}" >"${out}.log" 2>&1; then
            n_ok=$((n_ok + 1))
        else
            n_fail=$((n_fail + 1))
            echo "    FAILED, see ${out}.log"
        fi
    fi
done

echo
echo "=== done ==="
echo "  repaired : ${n_ok}/${#FAILED[@]}"
echo "  failed   : ${n_fail}"
echo "  logs     : ${OUT_DIR}"
[[ "${n_fail}" -eq 0 ]] || exit 1
