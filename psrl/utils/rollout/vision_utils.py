"""Vision utilities for multimodal rollout.

Provides ``pil_images_to_base64`` for serialising PIL images to JPEG-encoded
base64 data-URLs, used when forwarding images to the SMG
``/v1/chat/completions`` endpoint as ``image_url`` content parts.

All encoding is offloaded to the default asyncio thread-pool executor so
the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

psrl_logger = logging.getLogger(__name__)


def _encode_single_image(image: PILImage.Image, fmt: str = "JPEG") -> str:
    """Encode one PIL image to a base64 data-URL string (sync, CPU-bound)."""
    buf = io.BytesIO()
    # Palette modes must go via RGBA to apply the alpha channel correctly.
    if image.mode in ("P", "PA"):
        image = image.convert("RGBA").convert("RGB")
    elif image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    image.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


async def pil_images_to_base64(images: list[PILImage.Image]) -> list[str]:
    """Encode a list of PIL images to base64 data-URL strings asynchronously.

    Args:
        images: List of PIL.Image.Image objects.

    Returns:
        List of ``"data:image/jpeg;base64,<b64>"`` strings, one per image.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: [_encode_single_image(img) for img in images])
