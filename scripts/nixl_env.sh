#!/bin/bash
# Source this before running NIXL/nixl_ep code on any node of the cluster.
#
#   source scripts/nixl_env.sh
#
# Why this is needed rather than relying on the install: the nixl_ep extension lives on
# the shared conda filesystem, so every node sees the same .so -- but its RUNPATH was
# baked at build time and points at the builder's node-local /tmp build tree:
#
#   RUNPATH: ... :/tmp/nixl_check/build_ep/src/core: ...
#
# On any other node that path does not exist, so the import fails with
# "libnixl.so: cannot open shared object file". The runtime libraries and backend
# plugins are therefore staged on the shared filesystem and pointed at explicitly.
#
# Rebuilding NIXL per node would also work, but one shared copy keeps every node on
# byte-identical libraries -- a version skew between nodes in an RDMA path is the kind
# of fault that shows up as a wrong-bytes transfer rather than a clean error.

_PSRL_CONDA_LIB=/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/lib
_PSRL_NIXL_PLUGINS=/apdcephfs_zwfy10/share_303541817/lhy/nixl_runtime/plugins

export LD_LIBRARY_PATH="${_PSRL_CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export NIXL_PLUGIN_DIR="${_PSRL_NIXL_PLUGINS}"

# UCX needs the interface that carries RDMA on this cluster; the default selection
# picks a management NIC on some nodes and silently falls back to TCP.
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bond1}"

unset _PSRL_CONDA_LIB _PSRL_NIXL_PLUGINS
