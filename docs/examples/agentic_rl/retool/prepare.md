# ReTool Data & Sandbox Preparation

Before launching training you need to prepare two things: the **DAPO math parquets**
and the **SandboxFusion code-execution service** that every rollout worker will call
into over HTTP.

---

## 1. Data

Download the DAPO math parquets into `${PSRL_WORKSPACE}/data/dapo/`:

```
${PSRL_WORKSPACE}/data/dapo/
├── dapo-math-17k.parquet    # training set (BytedTsinghua-SIA/DAPO-Math-17k)
├── aime-2024.parquet        # validation set (optional)
└── aime-2025.parquet        # validation set (default)
```

`CustomRLHFDataset` (in `examples/retool/retool.py`) handles tokenization, caches the
post-mapping dataset under `cache_dir/processed_datasets/<md5>.parquet`, and routes
each row through the right `(problem, answer)` extractor based on the parquet name.

:::{tip}
The download/conversion pattern follows veRL's data-prep conventions, see
**→ [veRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)**
for the underlying parquet schema.
:::

---

## 2. SandboxFusion (code-execution service)

Every rollout worker emits Python `code_interpreter` tool calls that hit
`http://localhost:8080/run_code`. That endpoint is served by a cluster-wide
**SandboxFusion** Swarm service. Bringing it up is a two-step + smoke-test process.

### Step 1: Bake the image onto every node

```bash
DOCKERHUB_MIRROR=docker.m.daocloud.io \
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=/jizhicfs/lhy/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh

DOCKER_NODE_IPS=28.49.196.175:8,...,29.162.224.113:8 \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=/jizhicfs/lhy/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

`docker_install.sh` pulls `code_sandbox:server` via `skopeo` into a shared-FS tar.
`docker_copy.sh` fans the tar out to every node via `pssh` and runs `docker load`.

### Step 2: Deploy the SandboxFusion Swarm service

```bash
SANDBOX_NODE_IPS=28.49.196.175:8,...,29.162.224.113:8 \
SANDBOX_NODE_NUM=8 \
  bash examples/retool/sandbox_fusion/launch_service.sh
```

This will `docker swarm init` on the first host, `swarm join` everyone else, create
an overlay network, and `docker service create --publish published=8080,target=8080,mode=host`
so every node has a sandbox replica bound to `localhost:8080`.

### Step 3: Smoke test

On any node:

```bash
curl http://localhost:8080/run_code \
    -H 'Content-Type: application/json' \
    --data-raw '{"code": "print(2+3)", "language": "python"}'
```

Expected: `"status": "Success"`, `"stdout": "5\n"`. If this fails from a rollout node,
every episode will collapse to `-0.6` reward.

---

```{seealso}
Full bake / Swarm / troubleshooting docs:
[`examples/retool/docker_scripts/README.md`](https://github.com/lhy101/psrl/blob/main/examples/retool/docker_scripts/README.md)
and
[`examples/retool/sandbox_fusion/README.md`](https://github.com/lhy101/psrl/blob/main/examples/retool/sandbox_fusion/README.md).
```
