"""Fetch each handbook at a pinned commit, so the index and the eval set never drift apart."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import Workspace


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def current_commit(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=repo_dir)
    except subprocess.CalledProcessError:
        return None


def fetch_workspace(ws: Workspace, data_dir: Path) -> tuple[Path, bool]:
    """Sparse, shallow, blob-filtered checkout of just the handbook folder at ws.commit.

    Returns (directory, changed). Re-running is a no-op when the pinned commit is already there.
    """
    dest = data_dir / "corpora" / ws.id
    if current_commit(dest) == ws.commit:
        return dest / ws.path, False
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git("init", "-q", cwd=dest)
        _git("remote", "add", "origin", ws.repo, cwd=dest)
    _git("sparse-checkout", "set", ws.path, cwd=dest)
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ws.commit, cwd=dest)
    _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=dest)
    return dest / ws.path, True
