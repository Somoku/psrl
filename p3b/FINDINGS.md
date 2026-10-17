# Findings: the sandbox cgroup domain and the warm pool

Results for the two changes in `internal/cgroup`, `internal/pressure` and the docker
backend's warm pool. Each number names the configuration it was measured on; the scripts
that produce them are listed with it. Negative results and the limits of each claim are
at the end, because a claim without its scope is how a benchmark becomes folklore.

## Configuration measured

Two-node fleet, cgroup v1, Docker with the `cgroupfs` driver, both hosts 2319618 MB and
384 cores. Nodes declare 1100000 MB (the one that also runs the trainer) and 1400000 MB
(the sandbox-only one), `overcommit` 1.5, `local_mem_ceiling` 0.9, `utilization_target`
0.75, `cgroup_parent: psrl-sandboxes`, `sample_interval_s: 5`.

## 1. The measured domain is the sandbox subtree, not the pod

`internal/cgroup/live_test.go`, plus a fleet report before and after the change.

| Quantity | Before | After |
| --- | --- | --- |
| node reported `mem_used_pct` (the node that shares its cgroup with the trainer) | **48.8%** | **0.00%** |
| what that fraction describes | the pod: 1100 GiB of 2252 GiB, of which containers were 109 GiB | the subtree: 0 of 1650000 MiB |
| subtree limit resolved on that node | — | 1650000 MiB (requested 1100000 x 1.5) |
| subtree limit resolved on the sandbox-only node | — | 2041953 MiB, from a request of 2100000: capped by the ancestor, whose 2252800 MiB limit held 98206 MiB of unreclaimable memory plus a 5% safety share |

Exclusivity, measured by charging memory inside and outside the subtree:

| Charge | Subtree movement |
| --- | --- |
| 1 GiB container created with the daemon's default parent | 0 MiB (below the 512 MiB threshold the test sets) |
| 1 GiB container created with `--cgroup-parent psrl-sandboxes` | 1030 MiB |
| 4 GiB container created with `--cgroup-parent psrl-sandboxes` | 4108 MiB |

`Setup` also had to learn two facts the hard way, both caught by the live tests:

- `memory.memsw.limit_in_bytes` must be **raised before** `memory.limit_in_bytes` and
  lowered after it. Writing memsw down first returns `EINVAL`, because a fresh cgroup
  is unlimited in both and the kernel requires `memory <= memsw` at every step.
- Hierarchical accounting (`memory.use_hierarchy`) is verified rather than assumed. With
  it off a subtree reports only its own processes, so every ceiling derived from it is
  unreachable.

## 2. A warm claim costs milliseconds; a cold create costs hundreds

`tests/sandbox/bench_warm_pool.py`, against a node with `warm_pool` of 8 entries built
from `alpine:latest` (512 MB, 1 CPU, class `rollout`, one-shot, 7200 s lifetime).

Paired, alternating order, eight of each, one at a time:

| Kind | p50 | min | max |
| --- | --- | --- | --- |
| claimed from the pool | **0.004 s** | 0.004 s | 0.006 s |
| cold create + start | 0.274 s | 0.253 s | 0.520 s |

Sixteen concurrent, pool full (so eight of the sixteen claim) versus sixteen concurrent
that the pool cannot serve:

| Batch | p50 | p95 | min |
| --- | --- | --- | --- |
| 8 warm + 8 cold | 0.471 s | 0.894 s | 0.006 s |
| 16 cold | 0.831 s | 1.337 s | 0.588 s |

The batch improves by less than a single claim does, and that is the predicted shape: the
pool covers `size/batch` of a burst, so a depth of 8 halves a batch of 16. The earlier
create-contention work measured cold creates at 3.3 s with 8 in flight and 10.7 s with 64,
which is where a claim's 4 ms matters most -- this image is small enough that a cold
create on an idle daemon is only ~0.3 s, so these numbers understate the pool's value at
the workload's real concurrency rather than overstating it.

The measurements also found three defects that no unit test covered:

1. The gate's pressure sample is only refreshed by an admit or a report, so a node's very
   first pool refill saw an *unusable* sample and did nothing for one refill interval.
2. `Config.WorkingDir` in an inspect reports the *resolved* directory (the image's `/`),
   not the empty request, so verification rejected every entry whose caller stated no
   workdir -- and destroyed it. Claims looked like 1.1 s creates because they were
   destroys plus cold creates. This is why the "warm" batch was slower than the cold one
   in the first measurement.
3. A pool that cannot build its entries was silent: it simply never served a claim. It
   now warns with the create error.

## 3. Limits of these results

- **The pool serves one image, and this workload uses one image per task.** The mini_swe
  rollout resolves a per-problem base image (`psrl/swebench-harness:<digest>` of that
  problem's image), so a pool built from a single image can only serve the tasks whose
  image it names. That is the pool's design -- a pool across images divides a fixed
  budget by the number of images and stops being a pool for any of them -- and it means
  the mechanism is right but inapplicable to this rollout until the workload shares an
  image (or the pool becomes per-image with a budget per key).
- Lifetime is matched as `request <= template`, so a template states the longest lifetime
  the deployment runs. For the codex recipe that number is the episode budget plus the
  grading budget plus the lifetime slack, i.e. `7200 + 0 + 3600 = 10800` seconds.
- **Coverage is `size/batch`.** A burst larger than the pool is mostly cold, and the
  budget is what bounds the depth (`memory_mb` of the declared envelope, not a class).
- **The domain does not increase capacity by itself.** It makes the measurement describe
  sandboxes, which is what the elastic controller (§6) needs; admission is still
  reservation-based, so concurrency is still bounded by the declared envelope.
- **Only the docker backend on this host was measured**, and only cgroup v1 with the
  cgroupfs driver. The v2 path is covered by unit tests, not by a deployment.
- The RL-level metrics the design asks for -- successful rollouts/s, complete groups/s,
  resident byte-seconds during model waits -- are not measured here. They need an A/B of
  full training steps, not these microbenchmarks.

## Reproducing

```
# unit tests, no privileges
go test ./internal/cgroup/ ./internal/pressure/ ./internal/backend/dockerbackend/

# the domain, against a real daemon and cgroup hierarchy
go test -tags integration ./internal/cgroup/ ./internal/pressure/

# a node with a warm_pool block configured, then
python tests/sandbox/bench_warm_pool.py <control-endpoint> --batches 16 --pool-size 8
```
