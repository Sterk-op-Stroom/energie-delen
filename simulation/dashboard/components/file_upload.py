"""Handle browser file uploads: bytes → temp file → Path."""

from __future__ import annotations

import tempfile
from pathlib import Path


_temp_files: dict[str, Path] = {}  # role → current temp path


def bytes_to_tempfile(data: bytes, role: str) -> Path:
    """Write uploaded bytes to a named temp file and return its Path.

    The previous temp file for this role is unlinked on the next upload.
    The caller is responsible for not holding stale paths.

    Args:
        data: Raw Parquet file bytes from pn.widgets.FileInput.
        role: Label used to track the temp file (e.g. 'prosumer', 'production').
    """
    old = _temp_files.get(role)
    if old is not None:
        try:
            old.unlink(missing_ok=True)
        except OSError:
            pass

    tmp = tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f"{role}_", delete=False
    )
    tmp.write(data)
    tmp.close()
    path = Path(tmp.name)
    _temp_files[role] = path
    return path
