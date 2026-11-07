import os
import math
import re
import logging

logger = logging.getLogger("MavicVLM")

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sign(val: float) -> float:
    if val > 0:
        return 1.0
    elif val < 0:
        return -1.0
    return 0.0


def heading_to_cardinal(yaw_deg: float) -> str:
    headings = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((yaw_deg % 360) + 22.5) / 45) % 8
    return headings[idx]


def extract_number(text: str, default: float) -> float:
    """Extract the first number from text, requiring it to be followed by
    optional unit indicators to avoid matching ports/IPs."""
    import re
    # Match number optionally followed by m/meters/deg/degrees/°
    pattern = r'\\b\\d+(?:\\.\\d+)?(?:\\s*(?:m|meters|deg|degrees|°))?\\b'
    numbers = re.findall(pattern, text)
    if numbers:
        # Strip unit suffix if present
        num_str = numbers[0].split()[0]
        return float(num_str)
    return default


def is_frame_uniform(image_path: str, threshold: float = 0.98) -> bool:
    """Check if the frame is completely uniform (e.g. solid gray viewport)."""
    try:
        from PIL import Image
        if not os.path.exists(image_path):
            return True
        with Image.open(image_path) as img:
            gray = img.convert("L")
            hist = gray.histogram()
            max_pixels = max(hist)
            total_pixels = gray.width * gray.height
            if max_pixels / total_pixels >= threshold:
                return True
    except Exception as e:
        logger.warning(f"Failed to check image uniformity: {e}")
    return False


def sanitize_lmstudio_url(raw_url: str) -> str:
    """Sanitize LM Studio URL: enforce scheme, replace localhost with 127.0.0.1."""
    from urllib.parse import urlparse
    parsed = urlparse(raw_url.strip())
    if parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
    else:
        base = raw_url.split("/")[0]
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
    base = base.replace("localhost", "127.0.0.1")
    return f"{base}/v1/chat/completions"


# ---------------------------------------------------------------------------
# HTTP Dashboard Server
# ---------------------------------------------------------------------------

