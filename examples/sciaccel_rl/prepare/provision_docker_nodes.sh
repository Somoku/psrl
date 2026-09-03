#!/usr/bin/env bash
# provision_docker_nodes.sh — make Docker on a set of hosts able to build images
# here, in parallel, idempotently.
#
# A fresh node in this environment cannot build a Docker image at all:
#
#   1. There is no direct route to the internet. Registry pulls need Tencent's
#      mirror, and apt inside a build needs the corporate HTTP proxy.
#   2. buildkit does NOT inherit the shell's http_proxy. Only /root/.docker/config.json
#      `proxies.default` reaches a build, which is why a build fails at
#      `apt-get update` even when `curl` works in the same shell.
#   3. harbor's `network_mode = "no-network"` needs dockerd's egress-control support,
#      which in turn needs the registry mirror to be *loaded*. Editing daemon.json is
#      not enough: dockerd only re-reads it on SIGHUP, so a node whose config was
#      edited after dockerd started still reports `egress support: False` and every
#      trial dies at init.
#
# Symptom without this: image builds take ~3.5 hours (apt at ~17 kB/s through the
# proxy, often hitting the build timeout). With it: ~1 minute.
#
# Usage:
#   bash provision_docker_nodes.sh --hosts ${PSRL_WORKSPACE}/hosts/32GPUs
#   bash provision_docker_nodes.sh --hosts-list 28.49.195.154,28.49.55.40
#   bash provision_docker_nodes.sh --hosts <file> --check    # report only, change nothing
#
# Options:
#   --hosts FILE          Hosts file, one address per line; '#' and blanks ignored.
#   --hosts-list LIST     Comma-separated addresses, instead of a file.
#   --registry-mirror URL Registry mirror (default: https://mirror.ccs.tencentyun.com).
#   --proxy URL           HTTP(S) proxy for builds (default: http://star-proxy.oa.com:3128).
#   --apt-mirror URL      apt mirror asserted reachable from a build container
#                         (default: http://mirrors.tencentyun.com). Only verified here;
#                         it is applied by the sciaccel-rl Dockerfile generator.
#   --check               Report each host's state and exit non-zero if any needs work.
#   --no-verify           Skip the post-change build test (faster, less certain).
#   --timeout S           Per-host timeout in seconds (default: 600).
#   --user USER           ssh as USER.
#   -h | --help           Print this help.
#
# Safe to re-run: every step is a no-op when already correct, and dockerd is only
# reloaded when its live config actually differs from the target.

set -euo pipefail

HOSTS_FILE=""
HOSTS_LIST=""
REGISTRY_MIRROR="https://mirror.ccs.tencentyun.com"
PROXY_URL="http://star-proxy.oa.com:3128"
APT_MIRROR="http://mirrors.tencentyun.com"
CHECK_ONLY=0
VERIFY=1
TIMEOUT=600
SSH_USER=""

usage() { sed -n '2,49p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)            HOSTS_FILE="$2"; shift 2 ;;
        --hosts-list)       HOSTS_LIST="$2"; shift 2 ;;
        --registry-mirror)  REGISTRY_MIRROR="$2"; shift 2 ;;
        --proxy)            PROXY_URL="$2"; shift 2 ;;
        --apt-mirror)       APT_MIRROR="$2"; shift 2 ;;
        --check)            CHECK_ONLY=1; shift ;;
        --no-verify)        VERIFY=0; shift ;;
        --timeout)          TIMEOUT="$2"; shift 2 ;;
        --user)             SSH_USER="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${HOSTS_FILE}" ]]; then
    [[ -f "${HOSTS_FILE}" ]] || { echo "ERROR: --hosts file not found: ${HOSTS_FILE}" >&2; exit 2; }
    mapfile -t HOSTS < <(grep -Ev '^[[:space:]]*(#|$)' "${HOSTS_FILE}")
elif [[ -n "${HOSTS_LIST}" ]]; then
    IFS=',' read -r -a HOSTS <<< "${HOSTS_LIST}"
else
    echo "ERROR: pass --hosts FILE or --hosts-list LIST." >&2
    usage >&2
    exit 2
fi
[[ ${#HOSTS[@]} -gt 0 ]] || { echo "ERROR: no hosts to provision." >&2; exit 2; }

# The remote script. Written to a temp file and piped over ssh stdin rather than
# interpolated into a command line: ssh concatenates its arguments and the remote
# shell re-splits them, so embedding a multi-line script in an argument requires
# two consistent layers of quoting and breaks on the first stray quote.
REMOTE_SCRIPT="$(mktemp /tmp/provision_docker_remote.XXXXXX.sh)"
trap 'rm -f "${REMOTE_SCRIPT}"' EXIT

cat > "${REMOTE_SCRIPT}" <<'REMOTE_EOF'
#!/usr/bin/env bash
# Runs on each target host. Reads its parameters from the environment so nothing
# has to be quoted into a command line.
set -uo pipefail

MIRROR="${P_REGISTRY_MIRROR}"
PROXY="${P_PROXY_URL}"
APT_MIRROR="${P_APT_MIRROR}"
CHECK_ONLY="${P_CHECK_ONLY}"
VERIFY="${P_VERIFY}"

CHANGED=0
NEEDS=0
fail() { echo "FAIL: $*"; exit 1; }

command -v docker >/dev/null 2>&1 || fail "docker is not installed"
docker info >/dev/null 2>&1 || fail "dockerd is not responding"

# --- 1. daemon.json: registry mirror + network address pools ---
# (keeping any nvidia runtime already present)
DAEMON_JSON=/etc/docker/daemon.json
python3 - "$DAEMON_JSON" "$MIRROR" "$CHECK_ONLY" <<'PY'
import json, os, sys

path, mirror, check_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try:
    with open(path) as fh:
        cfg = json.load(fh)
except (OSError, ValueError):
    cfg = {}

problems = []

mirrors = cfg.get("registry-mirrors") or []
if mirror not in mirrors:
    problems.append(f"registry-mirrors={mirrors}")

# Docker's default pool is 172.17-172.31/16, i.e. only ~31 networks. Every harbor
# trial creates TWO (the task network and its egress-control sidecar network), so
# concurrency above ~15 trials dies with "could not find an available,
# non-overlapping IPv4 address pool among the defaults". Observed at 32 networks
# with 58 containers up. Carving /24s out of 172.16/12 and 10.128/9 yields
# thousands of networks instead.
WANT_POOLS = [
    {"base": "172.16.0.0/12", "size": 24},
    {"base": "10.128.0.0/9", "size": 24},
]
pools = cfg.get("default-address-pools") or []
if pools != WANT_POOLS:
    n = 0
    for p in pools:
        try:
            prefix = int(str(p.get("base", "")).split("/")[1])
            n += 2 ** (int(p.get("size", prefix)) - prefix)
        except (IndexError, ValueError, TypeError):
            pass
    problems.append(f"default-address-pools gives ~{n or 31} networks")

# harbor's GPU tasks need the nvidia runtime; never drop it if it is configured.
runtimes = cfg.get("runtimes") or {}
if "nvidia" in runtimes and not runtimes["nvidia"].get("path"):
    problems.append("nvidia runtime present but has no path")

if not problems:
    print("DAEMON_OK")
    sys.exit(0)
if check_only:
    print("DAEMON_NEEDS: " + "; ".join(problems))
    sys.exit(10)

if mirror not in mirrors:
    cfg["registry-mirrors"] = [mirror] + [m for m in mirrors if m != mirror]
cfg["default-address-pools"] = WANT_POOLS
os.makedirs(os.path.dirname(path), exist_ok=True)
if os.path.exists(path):
    os.replace(path, path + ".bak")
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
print("DAEMON_WROTE: " + "; ".join(problems))
PY
DAEMON_STATE=$?
if [[ "$DAEMON_STATE" -eq 10 ]]; then
    NEEDS=1
elif [[ "$DAEMON_STATE" -ne 0 ]]; then
    fail "could not update $DAEMON_JSON"
fi

# --- 2. Client config: the only proxy buildkit actually honors ---
CLIENT_JSON=/root/.docker/config.json
python3 - "$CLIENT_JSON" "$PROXY" "$CHECK_ONLY" <<'PY'
import json, os, sys

path, proxy, check_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try:
    with open(path) as fh:
        cfg = json.load(fh)
except (OSError, ValueError):
    cfg = {}

want = {"httpProxy": proxy, "httpsProxy": proxy, "noProxy": "localhost,127.0.0.1,::1"}
current = (cfg.get("proxies") or {}).get("default") or {}
if all(current.get(k) == v for k, v in want.items()):
    print("CLIENT_OK")
    sys.exit(0)
if check_only:
    print(f"CLIENT_NEEDS_PROXY (has {current or 'nothing'})")
    sys.exit(10)

cfg.setdefault("proxies", {})["default"] = want
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
print("CLIENT_WROTE")
PY
CLIENT_STATE=$?
if [[ "$CLIENT_STATE" -eq 10 ]]; then
    NEEDS=1
elif [[ "$CLIENT_STATE" -ne 0 ]]; then
    fail "could not update $CLIENT_JSON"
fi

# --- 3. Make dockerd actually load the config ---
# The step people miss: dockerd re-reads daemon.json only when signalled, so a node
# edited after startup keeps serving the old config and harbor's no-network mode
# stays unavailable.
#
# SIGHUP is enough for registry mirrors, but NOT for default-address-pools: those
# are read once when the network controller initializes, so widening the pool needs
# a full restart. Restarting is safe here only because no containers should be
# running -- it is refused otherwise rather than killing someone's work.
LIVE_MIRRORS="$(docker info --format '{{.RegistryConfig.Mirrors}}' 2>/dev/null || echo '[]')"
MIRROR_LIVE=0
case "$LIVE_MIRRORS" in *"$MIRROR"*) MIRROR_LIVE=1 ;; esac

# Probe the live pool by creating a throwaway network and reading its subnet.
POOL_LIVE=0
PROBE_NET="provision-pool-probe-$$"
if docker network create "$PROBE_NET" >/dev/null 2>&1; then
    PROBE_SUBNET="$(docker network inspect "$PROBE_NET" --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || true)"
    docker network rm "$PROBE_NET" >/dev/null 2>&1 || true
    # A /24 means the widened pool is live; the stock pool hands out /16.
    case "$PROBE_SUBNET" in */24) POOL_LIVE=1 ;; esac
else
    PROBE_SUBNET="(could not create probe network -- pool may be exhausted)"
fi

if [[ "$MIRROR_LIVE" -eq 1 && "$POOL_LIVE" -eq 1 ]]; then
    echo "LIVE_CONFIG_OK (probe subnet $PROBE_SUBNET)"
else
    NEEDS=1
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        [[ "$MIRROR_LIVE" -eq 1 ]] || echo "LIVE_MIRROR_STALE (live: $LIVE_MIRRORS)"
        [[ "$POOL_LIVE" -eq 1 ]] || echo "LIVE_POOL_NARROW (probe subnet $PROBE_SUBNET; needs dockerd RESTART)"
    else
        if [[ "$POOL_LIVE" -eq 0 ]]; then
            RUNNING="$(docker ps -q | wc -l)"
            if [[ "$RUNNING" -gt 0 ]]; then
                fail "$RUNNING container(s) running; refusing to restart dockerd. Stop them, then re-run."
            fi
            if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet docker 2>/dev/null; then
                systemctl restart docker || fail "systemctl restart docker failed"
            else
                # No init supervisor here (this host is not systemd-booted), so a
                # bare SIGTERM would stop dockerd with nothing to bring it back and
                # take the whole node's Docker down. Relaunch it explicitly instead.
                DOCKERD_PID="$(pgrep -o dockerd 2>/dev/null || true)"
                [[ -n "$DOCKERD_PID" ]] || fail "cannot find dockerd to restart"
                DOCKERD_BIN="$(command -v dockerd || echo /usr/bin/dockerd)"
                [[ -x "$DOCKERD_BIN" ]] || fail "dockerd binary not found; restart it manually"
                kill -TERM "$DOCKERD_PID" || fail "SIGTERM to dockerd $DOCKERD_PID failed"
                for _ in $(seq 1 30); do
                    sleep 2
                    pgrep -o dockerd >/dev/null 2>&1 || break
                done
                nohup setsid "$DOCKERD_BIN" >> /var/log/docker.log 2>&1 < /dev/null &
            fi
            for _ in $(seq 1 45); do
                sleep 2
                docker info >/dev/null 2>&1 && break
            done
            docker info >/dev/null 2>&1 || fail "dockerd did not come back after restart"
        else
            DOCKERD_PID="$(pgrep -o dockerd 2>/dev/null || true)"
            [[ -n "$DOCKERD_PID" ]] || fail "cannot find dockerd to reload"
            kill -HUP "$DOCKERD_PID" || fail "SIGHUP to dockerd $DOCKERD_PID failed"
            for _ in $(seq 1 20); do
                sleep 2
                LIVE_MIRRORS="$(docker info --format '{{.RegistryConfig.Mirrors}}' 2>/dev/null || echo '[]')"
                case "$LIVE_MIRRORS" in *"$MIRROR"*) break ;; esac
            done
        fi
        CHANGED=1

        # Re-verify both properties rather than assuming the signal worked.
        LIVE_MIRRORS="$(docker info --format '{{.RegistryConfig.Mirrors}}' 2>/dev/null || echo '[]')"
        case "$LIVE_MIRRORS" in
            *"$MIRROR"*) ;;
            *) fail "dockerd still lacks the mirror after reload (live: $LIVE_MIRRORS)" ;;
        esac
        if docker network create "$PROBE_NET" >/dev/null 2>&1; then
            PROBE_SUBNET="$(docker network inspect "$PROBE_NET" --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || true)"
            docker network rm "$PROBE_NET" >/dev/null 2>&1 || true
            case "$PROBE_SUBNET" in
                */24) echo "LIVE_CONFIG_RELOADED (probe subnet $PROBE_SUBNET)" ;;
                *) fail "address pool still narrow after restart (probe subnet $PROBE_SUBNET)" ;;
            esac
        else
            fail "could not create a probe network after restart"
        fi
    fi
fi

# --- 4. Report the capability harbor actually gates on ---
# harbor rejects network_mode="no-network" unless dockerd reports egress support,
# which is what makes repair/implementation tasks runnable at all.
if docker info 2>/dev/null | grep -qi "egress"; then
    docker info 2>/dev/null | grep -i "egress" | head -1 | sed 's/^[[:space:]]*/  /'
fi

# --- 4b. Pre-pull the base images every task Dockerfile starts FROM ---
# Without this, each task build issues its own registry metadata HEAD, and the mirror
# intermittently drops those: `dial tcp 169.254.0.51:443: i/o timeout` and bare `EOF`,
# even while sequential curls to the same URL return 200 three times in a row. It
# resolves to a link-local address (169.254.0.51), i.e. a node-local proxy, which is
# what makes it flaky under load rather than cleanly unreachable.
#
# Warming the task images does NOT cover this. A node with env_setup down to 15 s still
# lost 13 of 129 tasks to it, because the verifier base image is a separate FROM that no
# amount of env-image layer caching supplies. Pulling once per node removes the request
# entirely: a local image needs no registry round-trip.
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    for img in python:3.13-slim debian:bookworm-slim alpine:3.19; do
        if docker image inspect "$img" >/dev/null 2>&1; then
            echo "  BASE_IMAGE_PRESENT $img"
        elif timeout 180 docker pull -q "$img" >/dev/null 2>&1; then
            echo "  BASE_IMAGE_PULLED $img"
        else
            # Not fatal: the build can still fetch it, just less reliably.
            echo "  BASE_IMAGE_PULL_FAILED $img (task builds will retry via the mirror)"
        fi
    done
else
    for img in python:3.13-slim debian:bookworm-slim alpine:3.19; do
        docker image inspect "$img" >/dev/null 2>&1 \
            && echo "  BASE_IMAGE_PRESENT $img" \
            || echo "  BASE_IMAGE_MISSING $img"
    done
fi

# --- 5. Verify a real build can reach the network ---
if [[ "$VERIFY" -eq 1 && "$CHECK_ONLY" -eq 0 ]]; then
    BUILD_DIR="$(mktemp -d /tmp/dockerprobe.XXXXXX)"
    # Pulls through the mirror AND resolves the apt mirror from inside a build.
    printf 'FROM alpine:3.19\nRUN wget -q --timeout=20 -O /dev/null %s/ && echo probe-ok\n' \
        "$APT_MIRROR" > "$BUILD_DIR/Dockerfile"
    # Retry once. A dockerd that has just restarted answers `docker pull` fine while
    # buildkit's own registry client still fails its metadata HEAD with a bare `EOF`.
    # Observed on a freshly provisioned node: verification failed, `docker pull
    # alpine:3.19` succeeded seconds later, and the identical verification then passed.
    # Reporting that first failure as a config problem sends you hunting a proxy bug
    # that is not there.
    BUILD_OK=0
    for attempt in 1 2; do
        if timeout 300 docker build --no-cache -t docker-provision-probe:latest "$BUILD_DIR" >"$BUILD_DIR/out" 2>&1; then
            BUILD_OK=1
            break
        fi
        [[ "$attempt" -eq 1 ]] && sleep 20
    done
    if [[ "$BUILD_OK" -eq 1 ]]; then
        echo "BUILD_VERIFY_OK"
    else
        echo "BUILD_VERIFY_FAILED after 2 attempts, last lines:"
        tail -12 "$BUILD_DIR/out" | sed 's/^/    /'
        rm -rf "$BUILD_DIR"
        exit 1
    fi
    docker rmi -f docker-provision-probe:latest >/dev/null 2>&1 || true
    rm -rf "$BUILD_DIR"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    [[ "$NEEDS" -eq 0 ]] && echo "RESULT: already provisioned" || { echo "RESULT: needs provisioning"; exit 3; }
else
    [[ "$CHANGED" -eq 1 ]] && echo "RESULT: provisioned (dockerd reloaded)" || echo "RESULT: provisioned (no reload needed)"
fi
exit 0
REMOTE_EOF

echo "=== provision_docker_nodes ==="
echo "  hosts           : ${#HOSTS[@]} (${HOSTS[*]})"
echo "  registry mirror : ${REGISTRY_MIRROR}"
echo "  build proxy     : ${PROXY_URL}"
echo "  apt mirror      : ${APT_MIRROR} (verified, applied by the Dockerfile generator)"
echo "  mode            : $([[ "${CHECK_ONLY}" -eq 1 ]] && echo 'check only' || echo 'apply')"
echo "  build verify    : $([[ "${VERIFY}" -eq 1 && "${CHECK_ONLY}" -eq 0 ]] && echo yes || echo no)"
echo

SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o BatchMode=yes
    -o ConnectTimeout=15
)
[[ -n "${SSH_USER}" ]] && SSH_OPTS+=(-l "${SSH_USER}")

LOG_DIR="$(mktemp -d /tmp/provision_logs.XXXXXX)"
PIDS=()
declare -A HOST_BY_PID

# Fan out in parallel: each host's dockerd reload is independent, and serially
# reloading 16 nodes would take as long as the sum of their pulls.
for host in "${HOSTS[@]}"; do
    log="${LOG_DIR}/${host//[:\/]/_}.log"
    ssh "${SSH_OPTS[@]}" "${host}" \
        "P_REGISTRY_MIRROR=$(printf '%q' "${REGISTRY_MIRROR}") \
         P_PROXY_URL=$(printf '%q' "${PROXY_URL}") \
         P_APT_MIRROR=$(printf '%q' "${APT_MIRROR}") \
         P_CHECK_ONLY=${CHECK_ONLY} P_VERIFY=${VERIFY} \
         timeout ${TIMEOUT} bash -s" < "${REMOTE_SCRIPT}" > "${log}" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    HOST_BY_PID[${pid}]="${host}"
done

declare -A RC_BY_HOST
for pid in "${PIDS[@]}"; do
    rc=0
    wait "${pid}" || rc=$?
    RC_BY_HOST["${HOST_BY_PID[${pid}]}"]=${rc}
done

echo "--- per-host result ---"
n_ok=0
n_fail=0
n_needs=0
for host in "${HOSTS[@]}"; do
    rc="${RC_BY_HOST[${host}]:-?}"
    log="${LOG_DIR}/${host//[:\/]/_}.log"
    result="$(grep -E '^RESULT:' "${log}" 2>/dev/null | tail -1 || true)"
    case "${rc}" in
        0)
            printf '  %-18s OK    %s\n' "${host}" "${result#RESULT: }"
            n_ok=$((n_ok + 1)) ;;
        3)
            printf '  %-18s NEEDS PROVISIONING\n' "${host}"
            n_needs=$((n_needs + 1)) ;;
        *)
            printf '  %-18s FAIL  (rc=%s)\n' "${host}" "${rc}"
            sed 's/^/      /' "${log}" | tail -8
            n_fail=$((n_fail + 1)) ;;
    esac
done

echo
echo "  ok: ${n_ok}   needs-provisioning: ${n_needs}   failed: ${n_fail}   logs: ${LOG_DIR}"
if [[ "${n_fail}" -gt 0 ]]; then
    echo
    echo "A failing host usually means one of:"
    echo "  * dockerd not running            -> systemctl start docker"
    echo "  * no route to the proxy          -> curl -sS -x ${PROXY_URL} https://example.com"
    echo "  * mirror unreachable from there  -> the node may be in a different VPC"
    exit 1
fi
[[ "${n_needs}" -eq 0 ]] || exit 3
exit 0
