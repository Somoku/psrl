#!/bin/bash
# Refuse to start unless the Ray cluster forces train and rollout onto different nodes.
#
#   check_two_node_split.sh <train_gpus_per_node> <gen_instances> <val_instances>
#
# The fractions are the bundle sizes from `ResourcePoolManager`: training takes 0.9 of
# a device and colocated validation the remaining 0.1, while every rollout instance
# takes a whole device. Pools are STRICT_PACK and the trainer creates the train pool
# before the rollout pools, so the rollout instances land on another node exactly when
# they do not fit beside training.
set -euo pipefail

TRAIN_GPUS=$1
GEN_INSTANCES=$2
VAL_INSTANCES=$3

python3 - "$TRAIN_GPUS" "$GEN_INSTANCES" "$VAL_INSTANCES" <<'PY'
import os
import sys

import ray

train_gpus, gen_instances, val_instances = map(int, sys.argv[1:])
train_need = 0.9 * train_gpus
gen_need = float(gen_instances)
val_need = 0.1 * val_instances

# A driver running on a node that is not the Ray head has no local session to discover;
# honour RAY_ADDRESS there. On the head, "auto" remains the right default.
ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", logging_level="ERROR")
nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
per_node = {int(n["Resources"]["GPU"]) for n in nodes}
if len(nodes) < 2:
    raise SystemExit(f"REQUIRE two GPU nodes to split the roles, found {len(nodes)}.")
if len(per_node) != 1:
    raise SystemExit(f"REQUIRE the same GPU count on every node, found {sorted(per_node)}.")
gpus = per_node.pop()
if train_need + gen_need <= gpus + 1e-9:
    raise SystemExit(
        f"Training needs {train_need} and rollout {gen_need} GPUs, both fit on a "
        f"{gpus}-GPU node. Lower the per-node GPU count or widen the roles."
    )
if train_need + val_need > gpus + 1e-9:
    raise SystemExit(
        f"Training and validation need {train_need + val_need} GPUs, node has {gpus}."
    )
print(
    f"Two-node split verified: {len(nodes)} nodes, {gpus} GPUs each, "
    f"train {train_need}, rollout {gen_need}, validate {val_need}."
)
ray.shutdown()
PY
