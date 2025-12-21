# Adapted from slime/slime/utils/http_utils.py
import asyncio
import json
import logging
import os
import random
import socket

import httpx

psrl_logger = logging.getLogger(__name__)


def find_available_port(base_port: int):
    """Find an available port starting from base_port."""
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


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
_http_client: httpx.AsyncClient | None = None

# Maximum concurrency for the global HTTP client
_client_concurrency: int = 0


async def _post(client, url, payload, max_retries=60):
    """POST JSON payload with retries.

    Args:
        client: httpx.AsyncClient instance.
        url: URL to POST to.
        payload: JSON-serializable payload to send.
        max_retries: Maximum number of retries on failure.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await client.post(url, json=payload or {})
            response.raise_for_status()
            try:
                output = response.json()
            except json.JSONDecodeError:
                output = response.text
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            psrl_logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                psrl_logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


def init_http_client(server_concurrency: int, rollout_engine_num: int):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency

    _client_concurrency = server_concurrency * rollout_engine_num
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
        )


async def post(url, payload, max_retries=60):
    """POST JSON payload using the global HTTP client."""
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialized. Call psrl.utils.common.http_utils.init_http_client() "
            "once in this process (e.g., in each Ray actor/worker __init__) before calling post()."
        )
    return await _post(_http_client, url, payload, max_retries)


async def get(url):
    """GET JSON payload using the global HTTP client."""
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialized. Call psrl.utils.common.http_utils.init_http_client() "
            "once in this process (e.g., in each Ray actor/worker __init__) before calling get()."
        )
    response = await _http_client.get(url)
    response.raise_for_status()
    output = response.json()
    return output
