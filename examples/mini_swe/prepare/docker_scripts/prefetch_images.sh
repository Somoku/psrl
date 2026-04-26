#!/usr/bin/env bash
# prefetch_images.sh — Pull per-SWE-problem Docker images referenced by a PSRL
# parquet dataset.
#
# Default pull backend is `skopeo` (user-space, honours http(s)_proxy, streams
# directly into the local docker daemon via docker.sock, so dockerd itself does
# not need any proxy config).
#
# Usage:
#   source /jizhicfs/lhy/env/psrl.sh       # sets http_proxy for skopeo
#   bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
#       --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
#       --workers 4
#
# Options (exactly one of --parquet / --images / --only is required):
#   --parquet PATH    Read unique images from a PSRL parquet file
#                     (extra_info.sandbox_overrides.environment.image).
#   --images FILE     Read images from FILE, one per line (blank lines and
#                     lines starting with '#' are ignored). Use '-' for stdin.
#   --only CSV        Comma-separated inline list of images to pull, e.g.
#                     --only swebench/xxx:latest,swebench/yyy:latest
#   --workers N       Max concurrent pulls (default: 4).
#   --method M        Pull backend: skopeo|docker (default: skopeo if available,
#                     else docker).
#   --image-dir DIR   If set, save each image as a docker-archive tar under DIR
#                     (skopeo only). Without --load, images are NOT imported
#                     into the local dockerd (pure "download" mode).
#   --load            After pulling to --image-dir, `docker load` each tar into
#                     the local dockerd. Ignored when --image-dir is unset
#                     (because then skopeo already writes straight to the
#                     daemon).
#   --mirror HOST     Single mirror host, e.g. docker.m.daocloud.io. Short
#                     docker.io refs get rewritten to pull via the mirror.
#   --mirrors LIST    Comma-separated list of mirror hosts to try in order.
#                     If a pull fails on one mirror (e.g. blob 404 from lazy
#                     mirror cache) the script transparently falls back to the
#                     next, and finally to docker.io (disable with
#                     --no-direct-fallback).
#                     Example: --mirrors docker.1ms.run,dockerpull.org,hub.rat.dev
#   --no-direct-fallback
#                     Do not try plain docker.io after all mirrors fail. Useful
#                     on clusters where docker.io is firewalled.
#   --retries N       Retry count per (image, mirror) on transient failures
#                     like "unexpected EOF" / blob 404 (default: 3).
#   --log-dir DIR     Where to write per-image skopeo/docker logs. Defaults
#                     to `<prepare>/_prefetch_logs/` (one level up from this
#                     script's `docker_scripts/` directory).
#   --force           Re-pull even if the tar / local image already exists.
#   --dry-run         Print images that would be pulled without actually pulling.
#
# Behaviour matrix:
#   method=skopeo, no --image-dir               : skopeo copy docker://IMG -> docker-daemon:IMG
#   method=skopeo, --image-dir DIR              : skopeo copy docker://IMG -> docker-archive:DIR/FOO.tar  (no docker load)
#   method=skopeo, --image-dir DIR, --load      : same as above, then `docker load -i DIR/FOO.tar` + `docker tag`
#   method=docker                               : `docker pull IMG` (requires dockerd itself to reach the registry)
#
# The script deduplicates images, checks what is already cached (as tar or as a
# local docker image), and pulls the rest in parallel with a concurrency guard.
set -euo pipefail

PARQUET=""
IMAGES_FILE=""
ONLY_LIST=""
WORKERS=4
DRY_RUN=0
METHOD=""
IMAGE_DIR=""
DO_LOAD=0
FORCE=0
MIRROR_OVERRIDE=""
MIRRORS_LIST=""
NO_DIRECT_FALLBACK=0
RETRIES=3
LOG_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parquet)   PARQUET="$2";         shift 2 ;;
        --images)    IMAGES_FILE="$2";     shift 2 ;;
        --only)      ONLY_LIST="$2";       shift 2 ;;
        --workers)   WORKERS="$2";         shift 2 ;;
        --method)    METHOD="$2";          shift 2 ;;
        --image-dir) IMAGE_DIR="$2";       shift 2 ;;
        --load)      DO_LOAD=1;            shift ;;
        --mirror)    MIRROR_OVERRIDE="$2"; shift 2 ;;
        --mirrors)   MIRRORS_LIST="$2";    shift 2 ;;
        --no-direct-fallback) NO_DIRECT_FALLBACK=1; shift ;;
        --retries)   RETRIES="$2";         shift 2 ;;
        --log-dir)   LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        --force)     FORCE=1;              shift ;;
        --dry-run)   DRY_RUN=1;            shift ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Validate image source (exactly one of parquet/images/only).
SOURCES=0
[[ -n "$PARQUET"     ]] && SOURCES=$((SOURCES+1))
[[ -n "$IMAGES_FILE" ]] && SOURCES=$((SOURCES+1))
[[ -n "$ONLY_LIST"   ]] && SOURCES=$((SOURCES+1))
if [[ "$SOURCES" -ne 1 ]]; then
    echo "ERROR: exactly one of --parquet / --images / --only is required." >&2
    exit 1
fi

# Make sure http_proxy is set for skopeo (idempotent; ignore if the file is
# missing so the script is still usable outside this environment).
source "${PSRL_WORKSPACE:-$HOME}/env/psrl.sh" 2>/dev/null || true

# Default to skopeo when available — that is the method that works behind the
# user-space proxy on this cluster.
if [[ -z "$METHOD" ]]; then
    if command -v skopeo >/dev/null 2>&1; then
        METHOD=skopeo
    else
        METHOD=docker
    fi
fi

case "$METHOD" in
    skopeo|docker) ;;
    *) echo "ERROR: --method must be skopeo or docker (got: $METHOD)" >&2; exit 1 ;;
esac

# Build the list of mirrors to try. Priority:
#   1. --mirrors LIST  (comma-separated)
#   2. --mirror HOST   (single)
#   3. $DOCKERHUB_MIRROR env var
# An empty string in the list means "pull from docker.io directly" and is
# always appended at the end as the final fallback.
MIRRORS_ARR=()
if [[ -n "$MIRRORS_LIST" ]]; then
    IFS=',' read -r -a MIRRORS_ARR <<< "$MIRRORS_LIST"
elif [[ -n "$MIRROR_OVERRIDE" ]]; then
    MIRRORS_ARR=("$MIRROR_OVERRIDE")
elif [[ -n "${DOCKERHUB_MIRROR:-}" ]]; then
    MIRRORS_ARR=("$DOCKERHUB_MIRROR")
fi
if [[ "$NO_DIRECT_FALLBACK" -ne 1 ]]; then
    MIRRORS_ARR+=("")
fi
if [[ ${#MIRRORS_ARR[@]} -eq 0 ]]; then
    MIRRORS_ARR=("")  # at least one attempt (direct docker.io)
fi
# Keep the legacy single-mirror env var set to the first non-empty mirror,
# so helper functions that only read $DOCKERHUB_MIRROR still DWIM.
export DOCKERHUB_MIRROR="${MIRRORS_ARR[0]:-}"
# Encode the list as a newline-separated string for export to worker shells.
MIRRORS_ENC=$(printf '%s\n' "${MIRRORS_ARR[@]}")
export MIRRORS_ENC

if [[ -n "$IMAGE_DIR" ]]; then
    if [[ "$METHOD" != "skopeo" ]]; then
        echo "ERROR: --image-dir requires --method skopeo." >&2
        exit 1
    fi
    mkdir -p "$IMAGE_DIR"
fi

if [[ $DO_LOAD -eq 1 && -z "$IMAGE_DIR" ]]; then
    echo "WARN: --load has no effect without --image-dir (skopeo writes directly to dockerd in that mode)." >&2
fi

export METHOD IMAGE_DIR DO_LOAD FORCE RETRIES

# Per-image log dir. Default: `<prepare>/_prefetch_logs/` (one level up from
# this script's `docker_scripts/` dir). Override with --log-dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -n "$LOG_DIR_OVERRIDE" ]]; then
    LOG_DIR="$LOG_DIR_OVERRIDE"
else
    LOG_DIR="$PREPARE_DIR/_prefetch_logs"
fi
mkdir -p "$LOG_DIR"
export LOG_DIR

echo "=== prefetch_images ==="
if [[ -n "$PARQUET" ]]; then
    echo "  source  : parquet=$PARQUET"
elif [[ -n "$IMAGES_FILE" ]]; then
    echo "  source  : images=$IMAGES_FILE"
else
    echo "  source  : --only (inline)"
fi
echo "  method  : $METHOD"
echo "  workers : $WORKERS"
_m_pretty=""
for _m in "${MIRRORS_ARR[@]}"; do
    _m_pretty+="${_m:-docker.io}, "
done
echo "  mirrors : ${_m_pretty%, }"
unset _m _m_pretty
echo "  out-dir : ${IMAGE_DIR:-<dockerd>}"
echo "  log-dir : $LOG_DIR"
echo "  retries : $RETRIES"
echo "  load    : $DO_LOAD"
echo "  force   : $FORCE"
echo

if [[ -n "$PARQUET" ]]; then
    echo "--- reading parquet: $PARQUET ---"
    IMAGES=$(python - "$PARQUET" <<'EOF'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path)
images = set()
for row in df.itertuples():
    ei = row.extra_info
    if isinstance(ei, dict):
        so = ei.get("sandbox_overrides", {})
        img = so.get("environment", {}).get("image", "")
    else:
        img = ""
    if img:
        images.add(img)
for img in sorted(images):
    print(img)
EOF
)
elif [[ -n "$IMAGES_FILE" ]]; then
    if [[ "$IMAGES_FILE" == "-" ]]; then
        echo "--- reading images from stdin ---"
        raw=$(cat)
    else
        echo "--- reading images file: $IMAGES_FILE ---"
        raw=$(cat "$IMAGES_FILE")
    fi
    # Strip comments/blank lines, dedupe, keep stable order.
    IMAGES=$(printf '%s\n' "$raw" | sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | awk 'NF && !seen[$0]++')
else
    echo "--- reading images from --only ---"
    IMAGES=$(printf '%s\n' "$ONLY_LIST" | tr ',' '\n' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | awk 'NF && !seen[$0]++')
fi

# Normalize docker.io/ / index.docker.io/ prefixes so they dedupe cleanly with
# bare refs and produce consistent tar filenames (e.g. the Verified parquet
# often yields "docker.io/swebench/..." while the smith parquet yields bare
# "swebench/..."; we want both to map to the same cache key).
IMAGES=$(printf '%s\n' "$IMAGES" | sed -E 's#^(docker\.io|index\.docker\.io)/##' | awk 'NF && !seen[$0]++')

TOTAL=$(echo "$IMAGES" | grep -c . || true)
echo "Found $TOTAL unique images."

if [[ $DRY_RUN -eq 1 ]]; then
    echo "$IMAGES"
    echo "(dry-run, no images pulled)"
    exit 0
fi

# -----------------------------------------------------------------------------
# Helpers used by the parallel workers.
# -----------------------------------------------------------------------------

# Rewrite a short Docker Hub reference to use DOCKERHUB_MIRROR as the host.
# Strip the default-registry prefix. "docker.io/X" and "index.docker.io/X" both
# just mean "X on Docker Hub"; dropping the prefix lets the mirror rewriter
# treat them the same as bare refs (e.g. "swebench/foo").
normalize_image_ref() {
    local ref="$1"
    case "$ref" in
        docker.io/*)       ref="${ref#docker.io/}" ;;
        index.docker.io/*) ref="${ref#index.docker.io/}" ;;
    esac
    printf '%s' "$ref"
}

apply_dockerhub_mirror() {
    local ref="$1"
    local m="${DOCKERHUB_MIRROR:-}"
    # Normalize away docker.io/ / index.docker.io/ BEFORE deciding whether to
    # rewrite — otherwise a ref like "docker.io/swebench/foo:latest" would
    # short-circuit past the mirror and end up hitting registry-1.docker.io.
    ref=$(normalize_image_ref "$ref")
    [[ -n "$m" ]] || { printf '%s' "$ref"; return 0; }
    m="${m#/}"; m="${m%/}"
    case "$ref" in
        "$m"/*) printf '%s' "$ref"; return 0 ;;
    esac
    local first="${ref%%/*}"
    if [[ "$ref" == */* ]]; then
        if [[ "$first" == *.* || "$first" == localhost* ]]; then
            printf '%s' "$ref"
        else
            printf '%s/%s' "$m" "$ref"
        fi
    else
        printf '%s/library/%s' "$m" "$ref"
    fi
}

# Turn an image reference into a safe tar basename.
#   swebench/sweb.eval.x86_64.foo:latest  ->  swebench__sweb.eval.x86_64.foo__latest.tar
image_to_tar_name() {
    local img="$1"
    local name="${img//\//__}"
    name="${name//:/__}"
    printf '%s.tar' "$name"
}

# Detect whether this skopeo supports `--retry-times` (added in v1.2).
SKOPEO_RETRY_FLAG=""
if command -v skopeo >/dev/null 2>&1; then
    if skopeo copy --help 2>&1 | grep -q -- '--retry-times'; then
        SKOPEO_RETRY_FLAG="--retry-times 3"
    fi
fi
export SKOPEO_RETRY_FLAG

export -f normalize_image_ref apply_dockerhub_mirror image_to_tar_name

# Run a command with manual retry + proper exit-code handling. Stderr+stdout
# are tee'd to $2 so we keep a persistent log for debugging. Returns the
# underlying command's exit code (NOT tail/tee's).
run_with_log() {
    # usage: run_with_log LOG_FILE CMD...
    # Appends to LOG_FILE (does NOT truncate) so that logs accumulate across
    # retries and across mirror fall-backs. Retry count is $RETRIES (default 3).
    #
    # Optional env var $RW_PRE_CLEANUP — a path (e.g. a docker-archive tar)
    # that will be `rm -f`'d before EACH attempt. This is needed for
    # `skopeo copy docker-archive:<path>` because skopeo refuses to overwrite
    # an existing archive ("docker-archive doesn't support modifying existing
    # images"), so a failed partial write would poison every retry.
    local log="$1"; shift
    local attempt rc
    local n="${RETRIES:-3}"
    for (( attempt = 1; attempt <= n; attempt++ )); do
        if [[ -n "${RW_PRE_CLEANUP:-}" ]]; then
            rm -f "$RW_PRE_CLEANUP"
        fi
        {
            echo "----- attempt $attempt/$n @ $(date -Iseconds) : $* -----"
            if [[ -n "${RW_PRE_CLEANUP:-}" ]]; then
                echo "      (pre-attempt cleanup: rm -f $RW_PRE_CLEANUP)"
            fi
        } >> "$log"
        # pipefail needed so `cmd | tee` surfaces cmd's failure.
        ( set -o pipefail; "$@" 2>&1 | tee -a "$log" ) && return 0
        rc=$?
        echo "  [retry $attempt/$n] rc=$rc  (see $log)" >&2
        sleep $(( attempt * 2 ))
    done
    return "$rc"
}
export -f run_with_log

# Verify a docker-archive tar is structurally complete.
#   - non-empty
#   - valid tar stream (catches truncated / EOF / partial writes)
#   - contains a "manifest.json" entry (docker-archive marker)
# Returns 0 if valid, non-zero if missing/broken.
verify_docker_archive() {
    local tar="$1"
    [[ -s "$tar" ]] || return 1
    # `tar -tf` streams through the archive and fails on truncated data
    # ("Unexpected EOF in archive"). Grep short-circuits, but `tar -tf` keeps
    # reading and can set SIGPIPE rc; we look at PIPESTATUS[0] (tar's rc).
    local names
    names=$(tar -tf "$tar" 2>/dev/null) || return 1
    # Must contain the docker-archive manifest.
    grep -qx 'manifest.json' <<< "$names" || return 1
    return 0
}
export -f verify_docker_archive

# Truncate a log file and write a header. Use this at the start of each fresh
# pull_image invocation so that stale content from previous runs never leaks
# into the current log (run_with_log only appends).
init_log() {
    # usage: init_log LOG_FILE [HEADER_LINE...]
    local log="$1"; shift
    : > "$log"
    if [[ $# -gt 0 ]]; then
        printf '%s\n' "$@" >> "$log"
    fi
}
export -f init_log

# Rewrite IMG using a specific mirror host. Empty host means docker.io (no rewrite).
ref_for_mirror() {
    local img="$1" m="$2"
    DOCKERHUB_MIRROR="$m" apply_dockerhub_mirror "$img"
}
export -f ref_for_mirror

# Read the mirror list exported by the parent shell.
_mirrors_from_env() {
    local arr=()
    while IFS= read -r line; do arr+=("$line"); done <<< "$MIRRORS_ENC"
    printf '%s\n' "${arr[@]}"
}
export -f _mirrors_from_env

pull_image() {
    set -o pipefail
    local img="$1"
    local tar_name
    tar_name=$(image_to_tar_name "$img")
    local log_path="$LOG_DIR/${tar_name%.tar}.log"

    local tar_path=""
    if [[ -n "${IMAGE_DIR:-}" ]]; then
        tar_path="$IMAGE_DIR/$tar_name"
    fi

    # Integrity-aware cached short-circuit (skopeo + --image-dir only):
    #   - tar missing or empty            -> pull
    #   - tar present but corrupt/EOF'd   -> delete + pull (verify via tar -tf)
    #   - tar present and complete        -> skip, overwrite log with "already have"
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" && -e "$tar_path" && "$FORCE" -ne 1 ]]; then
        if verify_docker_archive "$tar_path"; then
            echo "  [cached-tar] $img  ($tar_path)"
            init_log "$log_path" \
                "========================================================" \
                "  已经拥有了 $img" \
                "  tar      : $tar_path ($(du -h "$tar_path" | cut -f1))" \
                "  verified : tar -tf passed, manifest.json present" \
                "  checked  : $(date -Iseconds)" \
                "========================================================"
            if [[ "$DO_LOAD" -eq 1 ]]; then
                if docker image inspect "$img" >/dev/null 2>&1; then
                    echo "  [loaded-cached] $img"
                    echo "(image already loaded into dockerd, skipping docker load)" >> "$log_path"
                    return 0
                fi
                echo "  [loading] $img  <-  $tar_path"
                init_log "$log_path.load" \
                    "========================================================" \
                    "  docker load $img  <-  $tar_path" \
                    "  started @ $(date -Iseconds)" \
                    "========================================================"
                if ! run_with_log "$log_path.load" docker load -i "$tar_path"; then
                    echo "  [FAILED]  $img (docker load; log: $log_path.load)" >&2
                    return 1
                fi
                echo "  [loaded]  $img"
            fi
            return 0
        else
            echo "  [corrupt] $img  ($tar_path) — deleting and re-pulling" >&2
            # Keep a note in the log before we overwrite it in the next step.
            local sz
            sz=$(du -h "$tar_path" 2>/dev/null | cut -f1)
            rm -f "$tar_path"
            # Fall through to the normal pull path. Log reset will include
            # a note about the corruption.
            CORRUPT_NOTE="previous tar was corrupt (size=$sz), removed and re-pulling"
        fi
    fi

    # Reset the main log file before real work so we do not mix stale content
    # from previous runs.
    init_log "$log_path" \
        "========================================================" \
        "  prefetch $img" \
        "  started @ $(date -Iseconds), method=$METHOD" \
        "========================================================"
    if [[ -n "${CORRUPT_NOTE:-}" ]]; then
        echo "  NOTE: $CORRUPT_NOTE" >> "$log_path"
        unset CORRUPT_NOTE
    fi

    # Try each mirror in order until one succeeds.
    local mirrors=()
    while IFS= read -r m; do mirrors+=("$m"); done < <(_mirrors_from_env)

    local mi=0 total_mirrors=${#mirrors[@]}
    echo "(attempting $total_mirrors mirror(s))" >> "$log_path"

    # Safety trap: if the pull phase is interrupted (Ctrl-C, SIGTERM, or any
    # uncaught error in this function), make sure we do NOT leave a partial
    # tar on disk. The trap is cleared once we either (a) complete successfully
    # so the tar is known-good, or (b) reach an explicit rm below.
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" ]]; then
        # On Ctrl-C / SIGTERM: wipe any partial tar, then actually exit the
        # worker. Without the explicit `exit`, bash would just resume the next
        # retry/mirror iteration, defeating the kill.
        # shellcheck disable=SC2064
        trap "rm -f '$tar_path'; echo '[interrupted] $img (tar cleaned)' >&2; exit 130" INT TERM
    fi

    for m in "${mirrors[@]}"; do
        mi=$((mi+1))
        local src
        src=$(ref_for_mirror "$img" "$m")
        local tag="[${mi}/${total_mirrors} mirror=${m:-docker.io}]"
        {
            echo
            echo "-------------------- $tag src=$src --------------------"
        } >> "$log_path"

        case "$METHOD" in
            skopeo)
                if [[ -n "$tar_path" ]]; then
                    echo "  [pulling] $img  $tag  ->  $tar_path"
                    # RW_PRE_CLEANUP deletes any partial tar BEFORE each retry,
                    # so skopeo won't bail out with
                    # "docker-archive doesn't support modifying existing images".
                    if RW_PRE_CLEANUP="$tar_path" run_with_log "$log_path" \
                            skopeo copy $SKOPEO_RETRY_FLAG \
                            "docker://$src" "docker-archive:$tar_path:$img"; then
                        echo "  [saved]   $img  ($tar_path)  via ${m:-docker.io}"
                        break
                    fi
                    rm -f "$tar_path"
                    echo "  (cleaned up partial tar $tar_path)" >> "$log_path"
                else
                    echo "  [pulling] $img  $tag  (skopeo -> docker-daemon)"
                    if run_with_log "$log_path" \
                            skopeo copy $SKOPEO_RETRY_FLAG \
                            "docker://$src" "docker-daemon:$img"; then
                        echo "  [done]    $img  via ${m:-docker.io}"
                        return 0
                    fi
                fi
                ;;
            docker)
                echo "  [pulling] $img  $tag  (docker pull $src)"
                if run_with_log "$log_path" docker pull "$src"; then
                    [[ "$src" != "$img" ]] && docker tag "$src" "$img"
                    echo "  [done]    $img  via ${m:-docker.io}"
                    return 0
                fi
                ;;
        esac
        echo "  [miss]    $img  via ${m:-docker.io}"
    done

    # If we get here in skopeo+tar mode, the tar was (or wasn't) produced by the
    # last successful iteration — check, verify integrity, then optionally load.
    if [[ "$METHOD" == "skopeo" && -n "$tar_path" ]]; then
        if [[ ! -s "$tar_path" ]] || ! verify_docker_archive "$tar_path"; then
            # Defensive: if any mirror thought it succeeded but the tar is
            # actually incomplete/corrupt, scrub it so future runs re-pull
            # instead of treating it as cached.
            if [[ -e "$tar_path" ]]; then
                echo "  (scrubbing incomplete/corrupt tar $tar_path)" >> "$log_path"
                rm -f "$tar_path"
            fi
            trap - INT TERM
            echo "  [FAILED]  $img (all mirrors failed; log: $log_path)" >&2
            return 1
        fi
        # Pull succeeded & tar verified — no more need for the cleanup trap.
        trap - INT TERM

        if [[ "$DO_LOAD" -eq 1 ]]; then
            echo "  [loading] $img  <-  $tar_path"
            init_log "$log_path.load" \
                "========================================================" \
                "  docker load $img  <-  $tar_path" \
                "  started @ $(date -Iseconds)" \
                "========================================================"
            if ! run_with_log "$log_path.load" docker load -i "$tar_path"; then
                # docker load failed but the tar is known-good (verified
                # above), so KEEP the tar for manual retry / inspection.
                echo "  [FAILED]  $img (docker load; tar kept at $tar_path; log: $log_path.load)" >&2
                return 1
            fi
            docker image inspect "$img" >/dev/null 2>&1 || \
                docker tag "$(ref_for_mirror "$img" "")" "$img" 2>/dev/null || true
            echo "  [loaded]  $img"
        fi
        return 0
    fi

    echo "  [FAILED]  $img (all ${total_mirrors} mirrors failed; log: $log_path)" >&2
    return 1
}
export -f pull_image

# -----------------------------------------------------------------------------
# Dedup / cache check.
# -----------------------------------------------------------------------------

is_cached() {
    local img="$1"
    [[ "$FORCE" -eq 1 ]] && return 1
    if [[ -n "${IMAGE_DIR:-}" ]]; then
        local tar_path="$IMAGE_DIR/$(image_to_tar_name "$img")"
        # Only treat as cached if the tar is structurally complete. A
        # truncated/EOF'd tar from a previous failed run must not be skipped.
        if verify_docker_archive "$tar_path"; then
            # If we also need it loaded, tar alone is not enough.
            if [[ "$DO_LOAD" -eq 1 ]]; then
                docker image inspect "$img" >/dev/null 2>&1
            else
                return 0
            fi
        else
            return 1
        fi
    else
        docker image inspect "$img" >/dev/null 2>&1
    fi
}

# Pre-scan purely for the human-readable summary banner. Every image — cached
# OR to-pull — is then dispatched through pull_image so each one gets a log
# file (the cache short-circuit in pull_image writes a "已经拥有了" log and
# exits quickly; no re-pull is done).
TO_PULL=()
CACHED_IMAGES=()
ALL_IMAGES=()
while IFS= read -r img; do
    [[ -z "$img" ]] && continue
    ALL_IMAGES+=("$img")
    if is_cached "$img"; then
        CACHED_IMAGES+=("$img")
    else
        TO_PULL+=("$img")
    fi
done <<< "$IMAGES"

echo
echo "Summary: ${#CACHED_IMAGES[@]} cached, ${#TO_PULL[@]} to pull, ${#ALL_IMAGES[@]} total (${WORKERS} workers)."
if [[ ${#ALL_IMAGES[@]} -eq 0 ]]; then
    echo "No images to process."
    exit 0
fi

if command -v parallel > /dev/null 2>&1; then
    printf '%s\n' "${ALL_IMAGES[@]}" | parallel --jobs "$WORKERS" pull_image {}
else
    printf '%s\n' "${ALL_IMAGES[@]}" | xargs -P "$WORKERS" -I{} bash -c 'pull_image "$@"' _ {}
fi

echo "=== prefetch_images: done ==="
