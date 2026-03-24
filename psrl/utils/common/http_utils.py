# Adapted from slime/slime/utils/http_utils.py
import asyncio
import json
import logging
import os
import socket
from typing import Any

import aiohttp

psrl_logger = logging.getLogger(__name__)


def find_available_port(base_port: int):
    """Find an available port starting from base_port."""
    port = base_port
    while not is_port_available(port):
        port += 1
    return port


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    """Get the hostname and local IP address of the current machine."""
    hostname = socket.gethostname()

    # try DNS
    try:
        return hostname, socket.gethostbyname(hostname)
    except socket.gaierror:
        pass

    # try IPv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.connect(("8.8.8.8", 80))  # Google DNS
            return hostname, udp_sock.getsockname()[0]
    except OSError:
        pass

    # try IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s6:
            s6.connect(("2001:4860:4860::8888", 80))
            return hostname, s6.getsockname()[0]
    except OSError:
        pass

    # hostname -I
    try:
        local_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
        return hostname, local_ip or "::1"
    except Exception:
        return hostname, "::1"


# Global HTTP client for POST/GET requests
_http_client: aiohttp.ClientSession | None = None

# Maximum concurrency for the global HTTP client
_client_concurrency: int = 256


async def _ensure_http_client() -> aiohttp.ClientSession:
    global _http_client
    if _http_client is None or _http_client.closed:
        connector = aiohttp.TCPConnector(
            limit=_client_concurrency,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        # Respect configured request timeout when provided.
        _http_client = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
        )
    return _http_client


async def _post(client, url, payload, max_retries=5, headers: dict[str, str] | None = None):
    """POST JSON payload with retries.

    Args:
        client: aiohttp.ClientSession instance.
        url: URL to POST to.
        payload: JSON-serializable payload to send.
        max_retries: Maximum number of retries on failure.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with client.post(url, json=payload or {}, headers=headers) as response:
                base_worker_id = response.headers.get("x-base-worker-id", None)
                target_dp_rank = response.headers.get("x-target-dp-rank", None)

                if response.status >= 400:
                    response_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=response_text,
                        headers=response.headers,
                    )
                try:
                    output = await response.json(content_type=None)
                    output["header_info"] = {
                        "base_worker_id": base_worker_id,
                        "target_dp_rank": target_dp_rank,
                    }
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    output = await response.text()
        except Exception as e:
            retry_count += 1
            psrl_logger.info(
                "Error: %s, retrying... (attempt %s/%s, url=%s)",
                e,
                retry_count,
                max_retries,
                url,
            )
            if retry_count >= max_retries:
                psrl_logger.info("Max retries (%s) reached, failing... (url=%s)", max_retries, url)
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


async def _get(client, url, params=None, max_retries=5, headers: dict[str, str] | None = None):
    """GET JSON payload with retries."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with client.get(url, params=params, headers=headers) as response:
                base_worker_id = response.headers.get("x-base-worker-id", None)
                target_dp_rank = response.headers.get("x-target-dp-rank", None)

                if response.status >= 400:
                    response_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=response_text,
                        headers=response.headers,
                    )
                try:
                    output = await response.json(content_type=None)
                    output["header_info"] = {
                        "base_worker_id": base_worker_id,
                        "target_dp_rank": target_dp_rank,
                    }
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    output = await response.text()
        except Exception as e:
            retry_count += 1
            psrl_logger.info(
                "Error: %s, retrying... (attempt %s/%s, url=%s)",
                e,
                retry_count,
                max_retries,
                url,
            )
            if retry_count >= max_retries:
                psrl_logger.info("Max retries (%s) reached, failing... (url=%s)", max_retries, url)
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


def init_http_client(server_concurrency: int, rollout_engine_num: int):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _client_concurrency

    _client_concurrency = server_concurrency * rollout_engine_num


async def post(url, payload, max_retries=5, headers: dict[str, str] | None = None):
    """POST JSON payload using the global HTTP client."""
    client = await _ensure_http_client()
    return await _post(client, url, payload, max_retries=max_retries, headers=headers)


async def get(url, params: dict[str, Any] | None = None, max_retries=5, headers: dict[str, str] | None = None):
    """GET JSON payload using the global HTTP client."""
    client = await _ensure_http_client()
    return await _get(client, url, params=params, max_retries=max_retries, headers=headers)
