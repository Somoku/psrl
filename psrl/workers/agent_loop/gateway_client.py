from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig
from verl import DataProto

from psrl.utils.common.http_utils import post
from psrl.utils.common.utils import b64_dumps, b64_loads


@dataclass(frozen=True)
class RolloutGatewayClient:
    """Small helper for calling RolloutGateway over HTTP.

    Payload contract:
    - Request body: {"dataproto_b64": <pickle+base64 DataProto>}
    - Response body: {"result": <pickle+base64 DataProto> | null}
    """

    base_url: str

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutGatewayClient:
        gw_cfg = config.psrl.server_rollout.gateway
        host = gw_cfg.get("router_ip", "127.0.0.1")
        port = int(gw_cfg.get("router_port", 18080))
        return cls(base_url=f"http://{host}:{port}")

    async def generate(self, request: DataProto) -> DataProto | None:
        payload = {"dataproto_b64": b64_dumps(request)}
        resp = await post(f"{self.base_url}/generate", payload)
        result_b64 = resp.get("result") if isinstance(resp, dict) else None
        if result_b64 is None:
            return None
        return b64_loads(result_b64)

    async def generate_async(self, request: DataProto) -> DataProto | None:
        payload = {"dataproto_b64": b64_dumps(request)}
        resp = await post(f"{self.base_url}/generate_async", payload)
        result_b64 = resp.get("result") if isinstance(resp, dict) else None
        if result_b64 is None:
            return None
        return b64_loads(result_b64)
