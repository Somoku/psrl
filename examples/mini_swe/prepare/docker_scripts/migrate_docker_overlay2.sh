#!/usr/bin/env bash
# migrate_docker_overlay2.sh — switch the (nested) dockerd from the `vfs`
# storage driver to `overlay2` on a copy-on-write filesystem.
#
# WHY
#   `vfs` gives every `docker create` a FULL copy of the image filesystem
#   (no CoW). With the ~3.8GB baked harness images this costs 7–25s per
#   sandbox, which dominates the harness `prep` phase (trajectory
#   `[Time Breakdown] sandbox=21–29s`). `overlay2` drops container creation
#   to <1s. The daemon currently uses `vfs` only because its data root lives
#   on `/` (an overlay mount) where overlay2 cannot nest.
#
#   This host exposes a dedicated XFS volume at /dockerdata (writable,
#   d_type-capable), so we move the daemon data root there with overlay2.
#
# MODES
#   migrate (default): export every image (by tag) -> restart dockerd on the
#                      new root with overlay2 -> re-import. Preserves all
#                      344 baked/base images and tags.
#   rebake            : restart with a FRESH overlay2 graph, then print the
#                       repopulation commands (prefetch/bake). Faster, but you
#                       must be able to re-fetch base images.
#
# SAFETY
#   * Refuses to run while any container is running (they will be killed by
#     the daemon restart) unless --force is given.
#   * Requires an explicit `YES` confirmation unless --yes is given.
#   * Never deletes the old vfs data root ($OLD_DATA_ROOT stays untouched) —
#     full rollback is just restarting dockerd with the old arguments.
#   * export/import are resumable (skips already-saved tars / existing images).
#
# USAGE
#   bash migrate_docker_overlay2.sh [--dry-run] [--mode migrate|rebake]
#                                  [--force] [--yes]
#
# ENV OVERRIDES
#   NEW_DATA_ROOT   new daemon data root   (default /dockerdata/docker)
#   NEW_EXEC_ROOT   new daemon exec root   (default /dockerdata/docker-exec)
#   NEW_PIDFILE     new daemon pidfile     (default /dockerdata/docker.pid)
#   NEW_LOG         new daemon log         (default /dockerdata/dockerd.log)
#   MIGRATION_DIR   export/import staging  (default /dockerdata/docker-migration)
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOCK="/var/run/docker.sock"
DAEMON_JSON="/etc/docker/daemon.json"
DAEMON_JSON_BAK="${DAEMON_JSON}.pre-overlay2.bak"

NEW_DATA_ROOT="${NEW_DATA_ROOT:-/dockerdata/docker}"
NEW_EXEC_ROOT="${NEW_EXEC_ROOT:-/dockerdata/docker-exec}"
NEW_PIDFILE="${NEW_PIDFILE:-/dockerdata/docker.pid}"
NEW_LOG="${NEW_LOG:-/dockerdata/dockerd.log}"
MIGRATION_DIR="${MIGRATION_DIR:-/dockerdata/docker-migration}"
REFS_FILE="${MIGRATION_DIR}/refs.txt"

MODE="migrate"
DRY_RUN=0
FORCE=0
ASSUME_YES=0

log()  { printf '[migrate] %s\n' "$*"; }
die()  { printf '[migrate] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[migrate] WARN: %s\n' "$*" >&2; }

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --mode) shift; MODE="${1:-migrate}" ;;
        --force) FORCE=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

case "$MODE" in
    migrate|rebake) ;;
    *) die "invalid --mode '$MODE' (expected migrate|rebake)" ;;
esac

# ---------------------------------------------------------------------------
# Current daemon facts
# ---------------------------------------------------------------------------
# The old vfs data root is inferred when the daemon is down; the previous
# daemon was launched with --data-root /tmp/psrl-docker-vfs/data.
OLD_DATA_ROOT_DEFAULT="/tmp/psrl-docker-vfs/data"

if docker info >/dev/null 2>&1; then
    DOCKER_OK=1
    CUR_DRIVER="$(docker info --format '{{.Driver}}')"
    CUR_DATA_ROOT="$(docker info --format '{{.DockerRootDir}}')"
    RUNNING_CONTAINERS="$(docker ps -q | wc -l)"
else
    DOCKER_OK=0
    CUR_DRIVER="vfs"                                  # inferred from old vfs data
    CUR_DATA_ROOT="$OLD_DATA_ROOT_DEFAULT"
    RUNNING_CONTAINERS=0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    printf '%s\n' "  => type YES to proceed, anything else to abort:" >&2
    read -r answer
    [ "$answer" = "YES" ]
}

image_count() { docker image ls -q | wc -l; }

image_total_bytes() {
    # Sum of `docker image ls` sizes (approximation of tar + new-graph cost).
    docker image ls --format '{{.Size}}' | python3 -c '
import sys, re
def b(s):
    m = re.match(r"^([\d.]+)\s*([kMGT]?B)?", s.strip())
    v = float(m.group(1))
    u = (m.group(2) or "B")[0]
    return v * {"": 1, "k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "B": 1}[u]
print(int(sum(b(l) for l in sys.stdin)))
'
}

free_bytes() {
    df -B1 --output=avail "$1" 2>/dev/null | tail -1
}

fs_type() { stat -f -c '%T' "$1" 2>/dev/null || echo unknown; }

# overlay2 needs d_type (ftype=1 on xfs). Best-effort probe: try a real
# overlay mount on the target FS; if we lack CAP_SYS_ADMIN, fall back to
# trusting that xfs/ext4 modern defaults provide ftype. dockerd itself is the
# final arbiter (verified right after restart).
overlay_supported() {
    local d="$1"
    local base lower upper work merged
    base="$(mktemp -d "$d/.ovlprobe.XXXXXX")"
    lower="$base/lower"; upper="$base/upper"; work="$base/work"; merged="$base/merged"
    mkdir -p "$lower" "$upper" "$work" "$merged"
    echo x > "$lower/probe"
    if mount -t overlay overlay -o "lowerdir=$lower,upperdir=$upper,workdir=$work" "$merged" 2>/dev/null; then
        ok=1; umount "$merged" 2>/dev/null || true
    else
        ok=0
    fi
    rm -rf "$base"
    return "$ok"
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight() {
    [ "$(id -u)" -eq 0 ] || die "must run as root (dockerd is root-owned)."

    # Target dirs are needed before we can probe the filesystem type / space.
    mkdir -p "$NEW_DATA_ROOT" "$NEW_EXEC_ROOT" "$MIGRATION_DIR/tars"

    log "daemon reachable=$DOCKER_OK driver=${CUR_DRIVER:-unknown} data-root=${CUR_DATA_ROOT:-unknown}"
    if [ "$DOCKER_OK" -eq 1 ]; then
        if [ "$CUR_DRIVER" = "overlay2" ] && [ "$CUR_DATA_ROOT" = "$NEW_DATA_ROOT" ]; then
            die "already migrated (overlay2 on $NEW_DATA_ROOT). Nothing to do."
        fi
        [ "$CUR_DRIVER" = "vfs" ] || warn "current driver is '$CUR_DRIVER' (expected vfs) — proceeding anyway."
    else
        warn "docker daemon is DOWN; assuming old vfs data at $CUR_DATA_ROOT."
        [ -d "$CUR_DATA_ROOT/vfs" ] || die "no vfs data found at $CUR_DATA_ROOT."
        if [ "$MODE" = "migrate" ]; then
            die "migrate mode needs the old daemon running to export images. Restore it first (original command), or use --mode rebake."
        fi
    fi

    command -v dockerd >/dev/null 2>&1 || die "dockerd binary not found on PATH."
    command -v setsid >/dev/null 2>&1 || die "setsid not available."

    # Target filesystem
    local ft
    ft="$(fs_type "$NEW_DATA_ROOT")"
    [ "$ft" = "xfs" ] || [ "$ft" = "ext2" ] || [ "$ft" = "ext3" ] || [ "$ft" = "ext4" ] \
        || die "target $NEW_DATA_ROOT is '$ft' — overlay2 needs xfs/ext4."
    [ -w "$NEW_DATA_ROOT" ] || die "target $NEW_DATA_ROOT is not writable."
    if ! overlay_supported "$NEW_DATA_ROOT"; then
        warn "overlay mount probe failed on $NEW_DATA_ROOT (may be a permission limitation)."
        warn "final verification happens right after the daemon restart."
    fi

    # Space: tars + new graph ≈ 2x image total (graph usually far less due to
    # shared layers on the SWE-bench bases).
    local total free
    total=0
    free="$(free_bytes "$NEW_DATA_ROOT")"
    if [ "$DOCKER_OK" -eq 1 ]; then
        total="$(image_total_bytes)"
        log "image total ~$(( total / 10**9 )) GB; free on $NEW_DATA_ROOT ~$(( free / 10**9 )) GB"
        if [ "$MODE" = "migrate" ] && [ "$free" -lt "$(( total + 50 * 10**9 ))" ]; then
            die "not enough free space on $NEW_DATA_ROOT for tars+graph."
        fi
    else
        log "free on $NEW_DATA_ROOT ~$(( free / 10**9 )) GB (daemon down; no export phase)"
    fi

    # Running containers / training guard
    if [ "$DOCKER_OK" -eq 1 ] && [ "$RUNNING_CONTAINERS" -gt 0 ]; then
        warn "there are $RUNNING_CONTAINERS running containers; the daemon restart WILL kill them."
        if [ "$FORCE" -ne 1 ]; then
            die "refusing to proceed with running containers (use --force to override)."
        fi
    fi
    if pgrep -f 'ray::|megatron|run_psrl|psrl\.' >/dev/null 2>&1; then
        warn "detected processes that look like a live training stack — double-check nothing is training."
    fi

    [ "$DRY_RUN" -eq 1 ] && return 0

    # Summary + confirmation
    local n
    if [ "$DOCKER_OK" -eq 1 ]; then
        n="$(image_count)"
    else
        n="unknown (daemon down)"
    fi
    printf '%s\n' \
        "=== PLAN ===" \
        "  daemon      : $(command -v dockerd)" \
        "  driver      : $CUR_DRIVER -> overlay2" \
        "  data-root   : $CUR_DATA_ROOT -> $NEW_DATA_ROOT" \
        "  mode        : $MODE" \
        "  images      : $n" \
        "  containers  : $RUNNING_CONTAINERS running (will be terminated)" \
        "  old data    : $CUR_DATA_ROOT is KEPT (rollback backup)" \
        "=== ACTION ==="
    confirm || die "aborted by user."
}

# ---------------------------------------------------------------------------
# Phase 1: export every image by tag (resumable)
# ---------------------------------------------------------------------------
export_images() {
    log "exporting $1 images to $MIGRATION_DIR/tars (resumable)..."
    : > "$REFS_FILE"
    local n=0
    docker image ls --format '{{.ID}}\t{{.Repository}}\t{{.Tag}}' | while IFS=$'\t' read -r id repo tag; do
        if [ "$repo" = "<none>" ] || [ "$tag" = "<none>" ]; then
            ref="$id"; fname="dangling_${id}.tar"
        else
            ref="${repo}:${tag}"; fname="$(printf '%s' "$ref" | tr '/:' '__').tar"
        fi
        printf '%s\t%s\n' "$fname" "$ref" >> "$REFS_FILE"
        if [ -f "$MIGRATION_DIR/tars/$fname" ]; then
            continue
        fi
        n=$((n + 1))
        log "  save ($n): $ref"
        docker save -o "$MIGRATION_DIR/tars/$fname.tmp" "$ref"
        mv "$MIGRATION_DIR/tars/$fname.tmp" "$MIGRATION_DIR/tars/$fname"
    done
    local saved
    saved="$(wc -l < "$REFS_FILE")"
    log "exported $saved image references."
}

# ---------------------------------------------------------------------------
# Phase 2: stop old daemon, start new daemon on overlay2
# ---------------------------------------------------------------------------
OLD_DAEMON_CMD=()
stop_dockerd() {
    log "stopping dockerd..."
    local pids first_pid
    pids="$(pgrep -x dockerd || true)"
    if [ -n "$pids" ]; then
        first_pid="$(printf '%s\n' $pids | head -1)"
        if [ -r "/proc/$first_pid/cmdline" ]; then
            mapfile -d '' -t OLD_DAEMON_CMD < "/proc/$first_pid/cmdline"
            log "captured old daemon cmdline for auto-rollback: ${OLD_DAEMON_CMD[*]}"
        fi
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
        local waited=0
        while kill -0 $pids 2>/dev/null && [ "$waited" -lt 60 ]; do
            sleep 1; waited=$((waited + 1))
        done
        if kill -0 $pids 2>/dev/null; then
            warn "dockerd did not exit after 60s; sending SIGKILL."
            # shellcheck disable=SC2086
            kill -KILL $pids 2>/dev/null || true
            sleep 2
        fi
    else
        warn "no dockerd process found (already stopped?)."
    fi
    rm -f "$SOCK"
    log "dockerd stopped."
}

restore_old_daemon() {
    if [ "${#OLD_DAEMON_CMD[@]}" -eq 0 ]; then
        warn "no old daemon cmdline captured; cannot auto-restore."
        return 1
    fi
    local old_log
    old_log="$(dirname "$CUR_DATA_ROOT")/dockerd.log"
    log "restoring old daemon: ${OLD_DAEMON_CMD[*]} (log: $old_log)"
    # shellcheck disable=SC2024
    setsid nohup "${OLD_DAEMON_CMD[@]}" >> "$old_log" 2>&1 < /dev/null &
    local waited=0
    until docker info >/dev/null 2>&1; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge 60 ]; then
            warn "old daemon did not become ready in 60s."
            return 1
        fi
    done
    log "old daemon restored (driver=$(docker info --format '{{.Driver}}'))."
    return 0
}

clear_daemon_json() {
    # Docker >= 25 hard-fails when the same directive is given BOTH as a CLI
    # flag and in /etc/docker/daemon.json (even with identical values). This
    # daemon is launched with explicit flags, so any daemon.json (e.g. one left
    # over from a previous attempt) must be removed — otherwise neither the new
    # overlay2 launch nor a rollback to the original vfs command will start.
    if [ -f "$DAEMON_JSON" ]; then
        if [ ! -f "$DAEMON_JSON_BAK" ]; then
            cp -a "$DAEMON_JSON" "$DAEMON_JSON_BAK"
            log "backed up $DAEMON_JSON -> $DAEMON_JSON_BAK"
        fi
        rm -f "$DAEMON_JSON"
        log "removed $DAEMON_JSON (daemon is flag-driven; avoids flag/file directive conflicts)."
    fi
}

start_dockerd() {
    clear_daemon_json
    log "starting dockerd (overlay2, data-root=$NEW_DATA_ROOT)..."
    # shellcheck disable=SC2024
    nohup setsid dockerd \
        --host "unix://$SOCK" \
        --data-root "$NEW_DATA_ROOT" \
        --exec-root "$NEW_EXEC_ROOT" \
        --pidfile "$NEW_PIDFILE" \
        --storage-driver overlay2 \
        >> "$NEW_LOG" 2>&1 < /dev/null &
    log "daemon launched; log: $NEW_LOG"

    local waited=0
    until docker info >/dev/null 2>&1; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge 120 ]; then
            log "daemon did not become ready in 120s. Tail of $NEW_LOG:"
            tail -30 "$NEW_LOG" >&2 || true
            if restore_old_daemon; then
                die "dockerd failed to start with overlay2; the old daemon was restored automatically. (see log tail above)"
            else
                die "dockerd failed to start AND the old daemon could not be auto-restored. Start dockerd manually (see log tail above)."
            fi
        fi
    done
    log "dockerd ready after ${waited}s."
}

# ---------------------------------------------------------------------------
# Phase 3: import (resumable)
# ---------------------------------------------------------------------------
import_images() {
    [ -f "$REFS_FILE" ] || die "no $REFS_FILE — run with --mode migrate."
    log "importing images (resumable)..."
    local n=0
    while IFS=$'\t' read -r fname ref; do
        [ -n "$fname" ] || continue
        tar="$MIGRATION_DIR/tars/$fname"
        if [ ! -f "$tar" ]; then
            warn "missing tar for $ref ($tar) — re-run export first."
            continue
        fi
        if docker image inspect "$ref" >/dev/null 2>&1; then
            continue
        fi
        n=$((n + 1))
        log "  load ($n): $ref"
        docker load -i "$tar"
        docker image inspect "$ref" >/dev/null 2>&1 \
            || die "image $ref did not appear after load."
    done < "$REFS_FILE"
    log "import complete."
}

# ---------------------------------------------------------------------------
# Phase 4: verify + report
# ---------------------------------------------------------------------------
verify() {
    local driver
    driver="$(docker info --format '{{.Driver}}' 2>/dev/null || true)"
    [ "$driver" = "overlay2" ] || die "driver is '$driver', expected overlay2."
    log "driver=$driver data-root=$(docker info --format '{{.DockerRootDir}}')"

    # Spot check: container boot from a real image.
    local ref probe
    ref="$(head -1 "$REFS_FILE" | cut -f2)"
    if [ -n "$ref" ]; then
        log "spot-check: docker run --rm $ref ..."
        probe="$( { time docker run --rm --entrypoint /bin/echo "$ref" ok; } 2>&1 | tr -d '\n' )" \
            || probe="spot-check failed (image may have a custom entrypoint; this is informational)"
        log "  -> $probe"
    fi
}

report_rollback() {
    printf '%s\n' \
        "" \
        "=== MIGRATION DONE ===" \
        "  driver      : $(docker info --format '{{.Driver}}' 2>/dev/null || echo '?')" \
        "  data-root   : $(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo '?')" \
        "  images      : $(image_count 2>/dev/null || echo '?')" \
        "" \
        "Old vfs data (backup, untouched): $CUR_DATA_ROOT" \
        "  -> delete it later only after you are confident, e.g.: rm -rf $CUR_DATA_ROOT" \
        "" \
        "ROLLBACK (restore vfs daemon):" \
        "  pkill -x dockerd; sleep 3" \
        "  nohup setsid dockerd --host unix:///var/run/docker.sock \\" \
        "    --data-root $CUR_DATA_ROOT --storage-driver vfs \\" \
        "    >> /tmp/psrl-docker-vfs/dockerd.log 2>&1 &" \
        "  (daemon.json is removed by this script, so the original vfs flags command works again.)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
preflight

if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run: preflight passed; no changes made. Run without --dry-run to migrate."
    exit 0
fi

if [ "$MODE" = "migrate" ]; then
    export_images "$(image_count)"
    stop_dockerd
    start_dockerd
    import_images
    verify
else
    stop_dockerd
    start_dockerd
    printf '%s\n' \
        "" \
        "=== REBAKE MODE ===" \
        "The overlay2 graph is fresh; repopulate images by re-running your usual prep:" \
        "  1) load base images, e.g.:" \
        "       bash prepare/docker_scripts/prefetch_images.sh ..." \
        "  2) re-bake harness derivatives:" \
        "       bash prepare/docker_scripts/bake_harness_image.sh --parquet <train.parquet>" \
        "  (any script that fails because the old vfs image is gone is expected.)"
fi

report_rollback
