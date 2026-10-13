"""Direct-supply geometry on real model shapes, computed on CPU.

Answers three questions the 5-node runs could only answer indirectly, using the
production planner (`CommunicationPlanner`) and the production sharding algebra
(`NIXLSharding.find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh` /
`get_local_sharded_tensors`) on meta tensors of the real shapes:

1. How many transfer descriptors does each read cost, upstream (train -> gen) and peer
   (gen -> gen)? Both are counted the way `client.py` registers them: one entry per
   finest-mesh cell, a strided cell expanding to one descriptor per row unless staged.
2. Would a peer read cost fewer descriptors if it moved the designate's *whole local
   tensor* instead of its finest-mesh cells? A follower with the same gen TP rank holds
   byte-identical cells, and on the gen side those cells tile a contiguous parameter.
3. How many bytes leave the trainer per sync, with and without peer diffusion, as the
   number of rollout nodes grows?

No GPU, no NIXL agent: this is a property of the plan and the layout.

    python tests/nixl/bench_direct_supply_geometry.py [--gen-nodes 2 4 8 16]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass

import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
)

BF16 = 2


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int


QWEN3_8B = ModelSpec("Qwen3-8B", 4096, 12288, 32, 8, 128, 36, 151936)
# 70B-class width, to show how the row count scales with hidden size.
LLAMA_70B = ModelSpec("Llama-70B-like", 8192, 28672, 64, 8, 128, 80, 128256)


def model_keys(m: ModelSpec) -> dict[str, tuple[tuple[int, ...], int | None]]:
    """``key -> (global shape, TP shard dim or None)`` in the unified HF namespace."""
    q = m.heads * m.head_dim
    kv = m.kv_heads * m.head_dim
    keys: dict[str, tuple[tuple[int, ...], int | None]] = {
        "model.embed_tokens.weight": ((m.vocab, m.hidden), 0),
        "model.norm.weight": ((m.hidden,), None),
        "lm_head.weight": ((m.vocab, m.hidden), 0),
    }
    for i in range(m.layers):
        p = f"model.layers.{i}."
        keys[p + "self_attn.q_proj.weight"] = ((q, m.hidden), 0)
        keys[p + "self_attn.k_proj.weight"] = ((kv, m.hidden), 0)
        keys[p + "self_attn.v_proj.weight"] = ((kv, m.hidden), 0)
        keys[p + "self_attn.o_proj.weight"] = ((m.hidden, q), 1)
        keys[p + "self_attn.q_norm.weight"] = ((m.head_dim,), None)
        keys[p + "self_attn.k_norm.weight"] = ((m.head_dim,), None)
        keys[p + "mlp.gate_proj.weight"] = ((m.intermediate, m.hidden), 0)
        keys[p + "mlp.up_proj.weight"] = ((m.intermediate, m.hidden), 0)
        keys[p + "mlp.down_proj.weight"] = ((m.hidden, m.intermediate), 1)
        keys[p + "input_layernorm.weight"] = ((m.hidden,), None)
        keys[p + "post_attention_layernorm.weight"] = ((m.hidden,), None)
    return keys


def layer_of(key: str) -> int | None:
    if key.startswith("model.layers."):
        return int(key.split(".")[2])
    return None


def rows_of(t: torch.Tensor) -> int | None:
    """Descriptor rows for a view, mirroring `strided_desc.describe_shard`.

    `describe_shard` refuses meta tensors (it needs a data pointer); the row count only
    depends on shape and stride, so it is recomputed here. None means not expressible as
    one strided range -- which none of these 2-D shapes hit.
    """
    sizes, strides = list(t.shape), list(t.stride())
    block = 1
    i = len(sizes) - 1
    while i >= 0 and strides[i] == block:
        block *= sizes[i]
        i -= 1
    if i < 0:
        return 1
    if any(sizes[o] != 1 for o in range(i)):
        return None
    return sizes[i]


@dataclass
class Client:
    name: str
    node: int
    gpu: int
    ctype: NIXLClientType
    # key -> (local sharding, local tensor shape)
    keys: dict[str, tuple[NIXLSharding, tuple[int, ...]]]


def local_shape(shape, dim, tp, rank):
    if dim is None:
        return tuple(shape)
    s = list(shape)
    assert s[dim] % tp == 0
    s[dim] //= tp
    return tuple(s)


def sharding_for(dim, tp, rank) -> NIXLSharding:
    if dim is None:
        return NIXLSharding.default()
    return NIXLSharding(shard_mesh=OrderedDict([(dim, tp)]), shard_indices=[tuple([rank])])


def build_cluster(m: ModelSpec, train_tp, train_pp, train_ranks, gen_tp, gen_nodes, gpus_per_node=8):
    keys = model_keys(m)
    clients: list[Client] = []
    layers_per_stage = m.layers // train_pp
    for r in range(train_ranks):
        tp, pp = r % train_tp, (r // train_tp) % train_pp
        owned = {}
        for key, (shape, dim) in keys.items():
            layer = layer_of(key)
            if layer is not None:
                if layer // layers_per_stage != pp:
                    continue
            elif key == "model.embed_tokens.weight" and pp != 0:
                continue
            elif key != "model.embed_tokens.weight" and pp != train_pp - 1:
                continue
            owned[key] = (sharding_for(dim, train_tp, tp), local_shape(shape, dim, train_tp, tp))
        clients.append(Client(f"train{r}", r // gpus_per_node, r % gpus_per_node, NIXLClientType.PUSH_SIDE, owned))
    train_nodes = -(-train_ranks // gpus_per_node)
    for i in range(gen_nodes * gpus_per_node):
        tp = i % gen_tp
        owned = {
            key: (sharding_for(dim, gen_tp, tp), local_shape(shape, dim, gen_tp, tp))
            for key, (shape, dim) in keys.items()
        }
        clients.append(
            Client(f"gen{i:03d}", train_nodes + i // gpus_per_node, i % gpus_per_node, NIXLClientType.PULL_SIDE, owned)
        )
    return clients, keys


def unify(clients: list[Client]) -> None:
    """`NIXLMetaServer.make_unified_sharding`: refactor every key to the global LCM mesh."""
    meshes = defaultdict(list)
    for c in clients:
        for key, (sh, _) in c.keys.items():
            meshes[key].append(sh.shard_mesh)
    finest = {k: NIXLSharding.find_finest_shard_mesh(v) for k, v in meshes.items()}
    for c in clients:
        for key, (sh, shape) in c.keys.items():
            sh = deepcopy(sh)
            sh.refactor_based_on_finer_shard_mesh(finest[key])
            c.keys[key] = (sh, shape)


def cells(sh: NIXLSharding, shape) -> list[torch.Tensor]:
    return sh.get_local_sharded_tensors(torch.empty(shape, dtype=torch.bfloat16, device="meta"))


def to_infos(clients: list[Client]) -> dict[str, NIXLClientInfo]:
    infos = {}
    for c in clients:
        tensor_infos = {}
        for key, (sh, shape) in c.keys.items():
            metas = [
                NIXLShardMetaInfo(
                    dtype=torch.bfloat16,
                    device=torch.device("cuda:0"),
                    shape=t.shape,
                    stride=t.stride(),
                    is_contiguous=t.is_contiguous(),
                )
                for t in cells(sh, shape)
            ]
            n = len(metas)
            tensor_infos[key] = NIXLTensorInfo(
                desc_bytes_list=[None] * n, temp_desc_bytes_list=[None] * n, sharding=sh, shard_meta_infos=metas
            )
        infos[c.name] = NIXLClientInfo(
            name=c.name,
            node_ip=f"10.0.{c.node}.1",
            node_gpu_id=c.gpu,
            type=c.ctype,
            tensor_infos=tensor_infos,
            meta=b"",
        )
    return infos


def analyse(m, train_tp, train_pp, train_ranks, gen_tp, gen_nodes, row_budget=64):
    clients, _ = build_cluster(m, train_tp, train_pp, train_ranks, gen_tp, gen_nodes)
    unify(clients)
    by_name = {c.name: c for c in clients}
    node_of = {c.name: c.node for c in clients}

    # Per (client, key, shard): (rows, bytes) of the registered cell.
    cell_rows: dict[tuple[str, str, tuple], int] = {}
    cell_bytes: dict[tuple[str, str, tuple], int] = {}
    whole_rows: dict[tuple[str, str], int] = {}
    for c in clients:
        for key, (sh, shape) in c.keys.items():
            whole_rows[(c.name, key)] = rows_of(torch.empty(shape, device="meta"))
            for idx, t in zip(sh.shard_indices, cells(sh, shape)):
                cell_rows[(c.name, key, idx)] = rows_of(t)
                cell_bytes[(c.name, key, idx)] = t.numel() * BF16

    infos = to_infos(clients)
    out = {}
    for diffusion in (False, True):
        plan = CommunicationPlanner(peer_diffusion=diffusion).make_comm_plan(infos)
        peer_plan = plan.peer_pull_plan
        gens = [c.name for c in clients if c.ctype is NIXLClientType.PULL_SIDE]

        # Which (gen, key, shard) are served by a peer: partition semantics, as the
        # pipelined harness runs them (the production loop never issues peer reads).
        peer_cover = set()
        for g, kp in peer_plan.items():
            for key, per_src in kp.items():
                for src, shards in per_src.items():
                    for s in shards:
                        peer_cover.add((g, key, s))

        egress_by_train_node = defaultdict(int)
        ingress_by_gen_node = defaultdict(int)
        up_desc_staged = up_desc_inplace = 0
        up_reads = 0
        for g in gens:
            for key, per_src in plan.rollout_pull_from_ps_plan.get(g, {}).items():
                for src, shards in per_src.items():
                    kept = [s for s in shards if (g, key, s) not in peer_cover]
                    if not kept:
                        continue
                    up_reads += 1
                    for s in kept:
                        b = cell_bytes[(g, key, s)]
                        egress_by_train_node[node_of[src]] += b
                        ingress_by_gen_node[node_of[g]] += b
                        gen_rows = cell_rows[(g, key, s)]
                        train_rows = cell_rows[(src, key, s)]
                        up_desc_inplace += max(gen_rows, train_rows)
                        staged = 1 if gen_rows > row_budget else gen_rows
                        up_desc_staged += max(staged, train_rows)

        peer_desc_cell = peer_desc_whole = 0
        peer_reads = peer_bytes = 0
        coalescible = total_peer_key_reads = 0
        for g, kp in peer_plan.items():
            for key, per_src in kp.items():
                for src, shards in per_src.items():
                    total_peer_key_reads += 1
                    peer_reads += 1
                    rows = sum(max(cell_rows[(g, key, s)], cell_rows[(src, key, s)]) for s in shards)
                    peer_desc_cell += rows
                    peer_bytes += sum(cell_bytes[(g, key, s)] for s in shards)
                    mine = set(by_name[g].keys[key][0].shard_indices)
                    theirs = set(by_name[src].keys[key][0].shard_indices)
                    if set(shards) == mine == theirs:
                        coalescible += 1
                        peer_desc_whole += max(whole_rows[(g, key)], whole_rows[(src, key)])
                    else:
                        peer_desc_whole += rows

        n_gen = len(gens)
        out["diffusion" if diffusion else "no_diffusion"] = {
            "gen_clients": n_gen,
            "upstream_reads": up_reads,
            "upstream_desc_per_gen_inplace": up_desc_inplace / n_gen,
            "upstream_desc_per_gen_staged": up_desc_staged / n_gen,
            "peer_reads": peer_reads,
            "peer_desc_per_gen_cells": peer_desc_cell / n_gen,
            "peer_desc_per_gen_whole_tensor": peer_desc_whole / n_gen,
            "peer_key_reads_coalescible": f"{coalescible}/{total_peer_key_reads}",
            "peer_GB": peer_bytes / 1e9,
            "trainer_egress_GB": sum(egress_by_train_node.values()) / 1e9,
            "max_egress_per_train_node_GB": max(egress_by_train_node.values()) / 1e9,
            "max_ingress_per_gen_node_GB": max(ingress_by_gen_node.values()) / 1e9,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-nodes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--model", choices=["qwen3-8b", "70b"], default="qwen3-8b")
    ap.add_argument("--train-tp", type=int, default=4)
    ap.add_argument("--train-pp", type=int, default=2)
    ap.add_argument("--train-ranks", type=int, default=24)
    ap.add_argument("--gen-tp", type=int, default=2)
    args = ap.parse_args()
    m = QWEN3_8B if args.model == "qwen3-8b" else LLAMA_70B
    results = {}
    for n in args.gen_nodes:
        results[n] = analyse(m, args.train_tp, args.train_pp, args.train_ranks, args.gen_tp, n)
        print(
            f"== {m.name} train TP={args.train_tp} PP={args.train_pp} x{args.train_ranks}, "
            f"gen TP={args.gen_tp}, gen nodes={n}"
        )
        print(json.dumps(results[n], indent=1))


if __name__ == "__main__":
    main()
