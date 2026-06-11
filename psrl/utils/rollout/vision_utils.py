import asyncio
import base64
import io
import logging

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


def extract_image_ref(part: dict):
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    if part_type in ("image", "input_image"):
        return part.get("image") or part.get("url") or part.get("image_url")
    return None


async def serialize_image_inputs(images: list) -> list[str]:
    if not images:
        return []

    urls: list[str | None] = []
    pil_images = []
    pil_positions = []
    for image in images:
        if isinstance(image, str):
            urls.append(image)
            continue
        if isinstance(image, dict):
            url = image.get("url")
            image_url = image.get("image_url")
            if url is None and isinstance(image_url, dict):
                url = image_url.get("url")
            elif url is None and isinstance(image_url, str):
                url = image_url
            if url is None:
                url = image.get("image") or image.get("path")
            if isinstance(url, str):
                urls.append(url)
                continue
        urls.append(None)
        pil_positions.append(len(urls) - 1)
        pil_images.append(image)

    if pil_images:
        encoded = await pil_images_to_base64(pil_images)
        for idx, value in zip(pil_positions, encoded, strict=False):
            urls[idx] = value
    return [url for url in urls if url is not None]
