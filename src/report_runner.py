"""Run a study's ``write_report()`` in a separate, low-priority, short-lived process.

The heavy number-crunching then never slows the 1-second samplers (no shared
GIL) and its memory is handed back to the server as soon as it finishes.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def run(module: str, tag: str, logger=None, timeout: float = 1200,
              env: Optional[dict] = None) -> bool:
    code = (f"from {module} import write_report; r = write_report(); "
            f"print(r.get('summary_line') or 'ok')")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, cwd=REPO, env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None)
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        if logger:
            logger.warning(f"{tag} report failed: took longer than {timeout:.0f}s")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"{tag} report failed: {e!r}")
        return False
    if proc.returncode != 0:
        if logger:
            last = (err or b"").decode("utf-8", "ignore").strip().splitlines()[-1:] or ["?"]
            logger.warning(f"{tag} report failed: {last[0]}")
        return False
    if logger:
        logger.info(f"{tag} report saved: {(out or b'').decode().strip()[:200]}")
    return True


def write_json(obj, out: str) -> None:
    import json
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, out)


def read_json(path: str) -> Optional[dict]:
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
