from __future__ import annotations

import base64
import json
import pathlib

from .context import get_plot_output_dir


def read_image(path: str) -> str:
    """
    Read an image from disk and return a JSON payload containing a data URL.
    """
    if not path:
        return "[READ_IMAGE ERROR] Missing required argument: path"

    root = pathlib.Path(get_plot_output_dir()).resolve()
    fpath = pathlib.Path(path).expanduser().resolve()
    try:
        fpath.relative_to(root)
    except ValueError:
        return f"[READ_IMAGE ERROR] Path not allowed (must be under {str(root)!r})."

    if not fpath.exists() or not fpath.is_file():
        return f"[READ_IMAGE ERROR] File not found: {str(fpath)!r}"

    ext = fpath.suffix.lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg", "webp"):
        return (
            "[READ_IMAGE ERROR] Unsupported image type (allowed: png, jpg, jpeg, webp)."
        )

    raw = fpath.read_bytes()
    max_bytes = 2_500_000
    if len(raw) > max_bytes:
        return f"[READ_IMAGE ERROR] Image too large ({len(raw)} bytes)."

    # Keep behavior compatible: always return a data URL. Use original mime type.
    mime = "image/png" if ext == "png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return json.dumps(
        {
            "path": str(fpath),
            "mime": mime,
            "size_bytes": len(raw),
            "data_url": f"data:{mime};base64,{b64}",
        }
    )
