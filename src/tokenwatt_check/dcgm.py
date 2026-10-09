"""NVIDIA DCGM diagnostics (`dcgmi diag`), when DCGM is installed on the node."""

from __future__ import annotations

import json
import shutil
import subprocess


def available() -> bool:
    return shutil.which("dcgmi") is not None


def diag(level: int = 1, timeout: int = 900) -> dict:
    """Run `dcgmi diag -r <level> -j`. Level 1 takes seconds, 2 about two minutes, 3 longer."""
    try:
        r = subprocess.run(["dcgmi", "diag", "-r", str(level), "-j"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"level": level, "result": "error", "error": f"{type(e).__name__}: {e}"[:200]}
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"level": level, "result": "error", "error": (r.stdout + r.stderr).strip()[:300]}
    failed: list[str] = []
    ran = 0

    def walk(node):
        nonlocal ran
        if isinstance(node, dict):
            name = node.get("name") or node.get("test_name")
            res = node.get("status") or node.get("result")
            if name and isinstance(res, str):
                ran += 1
                if res.lower() in ("fail", "failed", "error"):
                    failed.append(str(name))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(data)
    return {"level": level, "result": "fail" if failed else ("pass" if ran else "unknown"), "failed_tests": failed,
            "tests_seen": ran}
