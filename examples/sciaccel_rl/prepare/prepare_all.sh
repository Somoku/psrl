#!/usr/bin/env bash
# Run the SciAccel-RL preparation pipeline in dependency order.
#
# The four stages must happen in this sequence, and each one fails loudly rather
# than producing a half-usable artifact:
#
#   1. compile   authored task sources into Harbor tasks (build/<env>)
#   2. lines     resolve defect line numbers from pinned upstream source
#   3. dataset   build the L1/L2/L3 hinted parquets, train and val splits
#   4. warm      populate each node's buildkit cache so episodes do not cold build
#
# Order is not cosmetic. A dataset built before `compile` points at task
# directories with no `environment/`, which Harbor cannot build: every episode
# then dies about two seconds in with `unable to prepare context`. A dataset
# built before `lines` silently drops the line number from every L1 hint for an
# env that records only the file, which quietly turns L1 into L2.
#
# Stages are skippable so a re-run after a failure does not redo slow work:
#   prepare_all.sh --stages dataset,warm
#
# Usage:
#   prepare_all.sh [--repo PATH] [--envs a,b,c] [--hosts ip1,ip2]
#                  [--stages compile,lines,dataset,warm] [--difficulty easy]
#                  [--dry-run]

set -euo pipefail

PSRL_PATH=${PSRL_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}

REPO=${REPO:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/science_infra/sciaccel-rl}
# `laps` is stored already compiled and its tasks carry no difficulty field, so it
# is handled as a special case inside each stage rather than excluded here.
ENVS=${ENVS:-laps,mitgcm-biogeo,athena-gr}
HOSTS=${HOSTS:-28.58.246.40,28.59.83.117}
STAGES=${STAGES:-compile,lines,dataset,warm}
DIFFICULTY=${DIFFICULTY:-easy}
CATEGORIES=${CATEGORIES:-repair}
DATA_ROOT=${DATA_ROOT:-${PSRL_PATH}/examples/sciaccel_rl/data}
DRY_RUN=0

# Concurrent warm episodes per host, per environment. This is a per-env number
# because the envs differ by an order of magnitude in what one episode costs.
#
# `laps` compiles a small Fortran tree and its reference check runs in 1 to 6
# seconds, so 12 at once is comfortable: measured 99 of 99 tasks succeeding on two
# separate nodes.
#
# `mitgcm-biogeo` and `athena-gr` each compile a full scientific codebase, and
# `rowtool.py build --jobs 4` fans every episode out to 4 more processes. At 12 the
# result was a load average of 3616 with 1769 runnable threads on a 384 core node,
# which starved the containers' own 1 to 6 second reference runs past their 120 s
# `--strict` timeout and failed a third of the bank. The daemon stayed responsive
# throughout, so this is scheduler oversubscription rather than a Docker fault, and
# the fix belongs here rather than in the timeout.
WARM_CONCURRENCY_DEFAULT=${WARM_CONCURRENCY_DEFAULT:-4}
WARM_CONCURRENCY_LAPS=${WARM_CONCURRENCY_LAPS:-12}

warm_concurrency() {
    case "$1" in
        laps) echo "${WARM_CONCURRENCY_LAPS}" ;;
        *)    echo "${WARM_CONCURRENCY_DEFAULT}" ;;
    esac
}

usage() { sed -n '2,25p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)        REPO="$2"; shift 2 ;;
        --envs)        ENVS="$2"; shift 2 ;;
        --hosts)       HOSTS="$2"; shift 2 ;;
        --stages)      STAGES="$2"; shift 2 ;;
        --difficulty)  DIFFICULTY="$2"; shift 2 ;;
        --categories)  CATEGORIES="$2"; shift 2 ;;
        --data-root)   DATA_ROOT="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -d "${REPO}" ]] || { echo "ERROR: repo not found: ${REPO}" >&2; exit 2; }

IFS=',' read -r -a ENV_LIST <<< "${ENVS}"
IFS=',' read -r -a HOST_LIST <<< "${HOSTS}"

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}

# `laps` predates the difficulty tiers, so filtering it to `easy` matches nothing
# and `_discover_task_dirs` raises rather than silently emitting an empty dataset.
env_difficulty() {
    if [[ "$1" == "laps" ]]; then echo ""; else echo "${DIFFICULTY}"; fi
}

# The directory name records what the dataset actually is, because a repair-only
# easy bank and a mixed one are different experiments that must not share a path.
env_out_dir() {
    if [[ "$1" == "laps" ]]; then echo "${DATA_ROOT}/v2_repair"; else echo "${DATA_ROOT}/$1_${CATEGORIES}_${DIFFICULTY}"; fi
}

echo "=============================================================="
echo " repo    : ${REPO}"
echo " envs    : ${ENVS}"
echo " stages  : ${STAGES}"
echo " hosts   : ${HOSTS}"
echo "=============================================================="

# --- 1. Compile authored sources into Harbor tasks ---------------------------
# Skipped for an env that is already compiled in place, which is how `laps` ships.
if has_stage compile; then
    echo; echo "### [1/4] compile"
    for env in "${ENV_LIST[@]}"; do
        if [[ -f "${REPO}/build/${env}/index.jsonl" ]]; then
            echo "  ${env}: already compiled at build/${env}, skipping"
            continue
        fi
        if compgen -G "${REPO}/envs/${env}/tasks/*/*/environment" > /dev/null 2>&1 \
           || compgen -G "${REPO}/envs/${env}/tasks/*/environment" > /dev/null 2>&1; then
            echo "  ${env}: compiled in place, skipping"
            continue
        fi
        run python "${REPO}/utils/harbor/to_harbor.py" --env "${REPO}/envs/${env}"
    done
fi

# --- 2. Resolve defect line numbers -----------------------------------------
# Writes envs/<env>/factory/DEFECT_LINES.json. A recorded `candidate.meta.line`
# always wins and is never cached, so this only fills genuine gaps.
if has_stage lines; then
    echo; echo "### [2/4] resolve defect lines"
    args=()
    for env in "${ENV_LIST[@]}"; do args+=(--env "${env}"); done
    run python "${PSRL_PATH}/examples/sciaccel_rl/prepare/resolve_defect_lines.py" \
        --repo "${REPO}" "${args[@]}"
fi

# --- 3. Build the hinted datasets -------------------------------------------
if has_stage dataset; then
    echo; echo "### [3/4] build datasets"
    for env in "${ENV_LIST[@]}"; do
        out=$(env_out_dir "${env}")
        diff_arg=$(env_difficulty "${env}")
        cmd=(python -m examples.sciaccel_rl.prepare.build_dataset_v2
             --repo "${REPO}" --out-dir "${out}"
             --env "${env}" --categories "${CATEGORIES}" --hint-level all)
        [[ -n "${diff_arg}" ]] && cmd+=(--difficulty "${diff_arg}")
        ( cd "${PSRL_PATH}" && run "${cmd[@]}" )
    done
fi

# --- 4. Warm each node's image cache ----------------------------------------
# The `nop` agent builds the environment and edits nothing, so it populates
# /var/lib/docker/buildkit without a GPU or a served model. The cache is
# node-local, which is why every node that will host episodes needs its own pass.
if has_stage warm; then
    echo; echo "### [4/4] warm image caches"
    # Datasets are checked up front so a missing one fails before any host is touched.
    datasets=()
    for env in "${ENV_LIST[@]}"; do
        dataset="$(env_out_dir "${env}")/L1_all.parquet"
        if [[ ! -f "${dataset}" ]]; then
            echo "  ${env}: no dataset at ${dataset}, run the dataset stage first" >&2
            exit 2
        fi
        datasets+=("${env}:${dataset}")
    done

    for host in "${HOST_LIST[@]}"; do
        # A wedged daemon accepts the ssh and then hangs every build, so probe before
        # dispatching rather than discovering it 40 minutes later.
        if ! timeout 40 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" \
             'timeout 25 docker ps -q > /dev/null 2>&1'; then
            echo "  ${host}: docker is unresponsive, skipping (exclude it via AGENT_NODE_IPS)" >&2
            continue
        fi
        # One background shell per host walking the envs SEQUENTIALLY. Launching an
        # env per host in parallel would multiply the concurrency by the env count.
        remote_script=""
        for entry in "${datasets[@]}"; do
            env="${entry%%:*}"
            dataset="${entry#*:}"
            outdir="${PSRL_PATH}/examples/sciaccel_rl/outputs/warm/${env}_${host//./_}"
            conc=$(warm_concurrency "${env}")
            remote_script+="echo \"[warm] ${env} start \$(date +%T) conc=${conc}\"; "
            remote_script+="bash examples/sciaccel_rl/eval/run_eval.sh --agent nop "
            remote_script+="--dataset ${dataset} --output-dir ${outdir} "
            remote_script+="--skip-gpu-tasks --max-per-instance ${conc} -n ${conc} || true; "
        done
        remote_script+="echo \"[warm] all envs done \$(date +%T)\";"
        echo "  ${host}: ${#datasets[@]} envs, sequential"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            # `-n` and the redirects matter: without them ssh keeps the channel open
            # waiting on the remote's inherited stdout, so this loop blocks on host 1
            # and never dispatches host 2.
            ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" \
                "cd ${PSRL_PATH} && nohup setsid bash -c '${remote_script}' > /tmp/warm_all.log 2>&1 < /dev/null & disown" \
                > /dev/null 2>&1 </dev/null
        fi
        # Staggered on purpose: launching every host at once has raced and silently
        # dropped one, which then cold builds during training.
        sleep 2
    done
    echo
    echo "Warm runs in the background, roughly 40 to 60 minutes per env per host. Watch with:"
    for host in "${HOST_LIST[@]}"; do
        echo "  ssh ${host} 'tail -2 /tmp/warm_all.log; timeout 20 docker ps -q | wc -l'"
    done
fi

echo; echo "Done: ${STAGES}"
