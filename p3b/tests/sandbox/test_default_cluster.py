"""Start the shipped example deployment and drive it through the SDK.

This is the end-to-end check of the default topology: the two example node
agents and the example control plane, with the Docker backend, on the ports and
cgroup subtrees the repository ships. It is opt-in because it starts real
processes and needs a Docker daemon.

    PSRL_RUN_CLUSTER_INTEGRATION=1 pytest p3b/tests/sandbox/test_default_cluster.py -q

Ports and cgroup parents are rewritten per run, so the test does not collide with
a running deployment or with a previous run's leftovers.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from sandboxd import Resources, SandboxClient, SandboxSpec, Source

pytestmark = pytest.mark.skipif(
    os.environ.get("PSRL_RUN_CLUSTER_INTEGRATION") != "1",
    reason="starts real node and control processes; set PSRL_RUN_CLUSTER_INTEGRATION=1",
)

P3B = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("PSRL_TEST_IMAGE", "alpine:latest")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, deadline_s: float = 30.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def binary() -> str:
    override = os.environ.get("PSRL_SANDBOXD_BINARY")
    if override:
        return override
    target = Path(os.environ.get("TMPDIR", "/tmp")) / f"sandboxd-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(["go", "build", "-o", str(target), "./cmd/sandboxd"], cwd=P3B, check=True)
    return str(target)


def rewrite(config: dict, *, node_addresses: dict[str, str], cgroup_suffix: str) -> dict:
    config = json.loads(json.dumps(config))
    # The node TTL also sets the report interval (TTL/4) and must stay above the
    # placement reservation TTL, so the test leaves it at the shipped 120s and
    # waits for the first report instead of shortening it below a valid ordering.
    config.setdefault("timing", {})["rpc_timeout_s"] = 10
    if config.get("role") == "control":
        config["listen"] = f"127.0.0.1:{free_port()}"
        config["fleet_nodes"] = [
            {"node_id": node_id, "address": address}
            for node_id, address in node_addresses.items()
        ]
    else:
        node_id = config["node_id"]
        port = free_port()
        node_addresses[node_id] = f"127.0.0.1:{port}"
        config["node_listen"] = f"127.0.0.1:{port}"
        config["node"]["cgroup_parent"] = f"psrl-test-{cgroup_suffix}-{node_id}"
    return config


@pytest.mark.asyncio
async def test_the_shipped_example_deployment_places_and_executes(tmp_path: Path) -> None:
    if os.geteuid() != 0:
        pytest.skip("the service creates a cgroup subtree, which needs root")
    if not Path("/var/run/docker.sock").exists():
        pytest.skip("no Docker socket")
    if shutil.which("go") is None and not os.environ.get("PSRL_SANDBOXD_BINARY"):
        pytest.skip("go is required to build sandboxd")
    inspected = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, text=True
    )
    if inspected.returncode != 0:
        pytest.skip(f"test image {IMAGE} is not present")

    exe = binary()
    node_addresses: dict[str, str] = {}
    suffix = uuid.uuid4().hex[:8]
    processes: list[subprocess.Popen] = []
    logs: list[Path] = []
    configs: list[dict] = []
    control_port = 0

    def start(config: dict, name: str) -> subprocess.Popen:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config, indent=2))
        subprocess.run([exe, "-preflight", "-config", str(path)], check=True,
                       capture_output=True, text=True)
        log = tmp_path / f"{name}.log"
        log.touch()
        process = subprocess.Popen(
            [exe, "-config", str(path)],
            stdout=log.open("w"), stderr=subprocess.STDOUT,
        )
        processes.append(process)
        logs.append(log)
        configs.append(config)
        return process

    try:
        for name in ("example_node.json", "example_node_b.json"):
            node_config = rewrite(json.loads((P3B / name).read_text()),
                                  node_addresses=node_addresses, cgroup_suffix=suffix)
            start(node_config, Path(name).stem)

        control = rewrite(json.loads((P3B / "example_control.json").read_text()),
                          node_addresses=node_addresses, cgroup_suffix=suffix)
        control_port = int(control["listen"].rsplit(":", 1)[1])
        start(control, "control")

        assert wait_for_port(control_port), f"control did not bind\n{logs[-1].read_text()}"
        for address in node_addresses.values():
            assert wait_for_port(int(address.rsplit(":", 1)[1])), address

        async with SandboxClient(f"127.0.0.1:{control_port}") as client:
            deadline = time.monotonic() + 45
            report = await client.fleet()
            while len(report.nodes) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(1)
                report = await client.fleet()
            assert len(report.nodes) == 2, report

            sandboxes = await asyncio.gather(
                *(client.create(SandboxSpec(
                    source=Source.image(IMAGE),
                    resources=Resources(memory_mb=256, cpu_count=1),
                    resource_class="rollout",
                    lifetime_s=120,
                )) for _ in range(4))
            )
            try:
                assert {sandbox.node_id for sandbox in sandboxes} == set(node_addresses)
                for sandbox in sandboxes:
                    result = await sandbox.exec("echo cluster")
                    assert result.exit_code == 0
                    assert result.stdout.strip() == "cluster"
            finally:
                for sandbox in sandboxes:
                    await sandbox.release()
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_tail = "\n".join(
            f"--- {log.name} ---\n{log.read_text()[-2000:]}" for log in logs if log.exists()
        )
        for config in configs:
            parent = (config.get("node") or {}).get("cgroup_parent")
            if not parent:
                continue
            for root in Path("/sys/fs/cgroup").glob(f"*/{parent}"):
                for child in sorted(
                    (path for path in root.rglob("*") if path.is_dir()), reverse=True
                ):
                    try:
                        child.rmdir()
                    except OSError:
                        pass
                try:
                    root.rmdir()
                except OSError:
                    pass
        # Do not raise here: a cleanup-time raise would replace the assertion that
        # actually failed. The logs are printed so a startup or runtime fault is
        # visible in the failure output. A SIGKILL after the graceful-stop window
        # is expected for a control plane waiting on an in-flight node call.
        if any(process.returncode not in (0, -9, -15) for process in processes):
            print(f"a process exited unexpectedly\n{log_tail}")
