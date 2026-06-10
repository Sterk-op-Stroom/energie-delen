"""Handle browser file uploads: bytes → temp file → Path."""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

_cleanup_paths: set[Path] = set()


def register_cleanup(path: Path) -> None:
    """Register a file or directory to be deleted at process exit.

    Safe to call multiple times with the same path. Also safe to call for
    paths that no longer exist at exit time (e.g. already removed by the
    trash button).
    """
    _cleanup_paths.add(path)


def bytes_to_tempfile(data: bytes, role: str) -> Path:
    """Write uploaded bytes to a named temp file and return its Path.

    The path is automatically registered for deletion at process exit.
    It is also unlinked immediately when the user removes the row via the
    trash button — double-deletion is handled gracefully.

    Args:
        data: Raw Parquet file bytes from pn.widgets.FileInput.
        role: Label used as filename prefix (e.g. 'prosumer', 'production').
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f"{role}_", delete=False
    )
    tmp.write(data)
    tmp.close()
    path = Path(tmp.name)
    register_cleanup(path)
    return path


def resolve_role_path(
    files: list[tuple[Path, str]],
    indices: list[int],
    role: str,
    fallback: Path | None = None,
) -> Path | None:
    """Resolve the effective data path for a role from the current checkbox selection.

    Args:
        files:    List of (temp_path, original_filename) for all uploaded files of this role.
        indices:  Currently selected indices into *files*.
        role:     Label used as dir/file prefix ('prosumer' or 'production').
        fallback: Path to return when no files are selected (e.g. a manually typed path).

    Returns:
        - None if nothing is selected and fallback is None.
        - The single file's Path when exactly one file is selected.
        - A fresh temporary directory path containing copies of all selected files
          when more than one file is selected (the loader auto-discovers .parquet files).
    """
    selected = [files[i] for i in sorted(indices) if i < len(files)]
    if not selected:
        return fallback
    if len(selected) == 1:
        return selected[0][0]

    tmpdir = Path(tempfile.mkdtemp(prefix=f"{role}_merged_"))
    register_cleanup(tmpdir)
    for j, (path, _filename) in enumerate(selected):
        shutil.copy2(str(path), str(tmpdir / f"{role}_{j:03d}.parquet"))
    return tmpdir


def _cleanup_all() -> None:
    for p in list(_cleanup_paths):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_all)
