"""Host checks that decide how long a model takes to load: disk throughput and download speed."""

from __future__ import annotations

import os
import platform
import shutil
import time
import urllib.request

# a large public file on the Hugging Face CDN (what model weights come from)
DOWNLOAD_URL = "https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/resolve/main/model-00001-of-00007.safetensors"


def info() -> dict:
    out = {"hostname": platform.node(), "os": platform.platform(), "python": platform.python_version(),
           "cpus": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
        out["ram_gb"] = round(kb / 2 ** 20)
    except (OSError, StopIteration, ValueError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            out["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return out


def disk(path: str = ".", gib: float = 2.0) -> dict:
    """Sequential write (with fsync) and read of a temporary file in `path`."""
    usage = shutil.disk_usage(path)
    out = {"path": os.path.abspath(path), "free_gb": round(usage.free / 1e9)}
    if usage.free < gib * 2 ** 30 * 1.5:
        out["skipped"] = "not enough free space"
        return out
    fn = os.path.join(path, f".tokenwatt-check-{os.getpid()}.tmp")
    block = os.urandom(16 * 2 ** 20)
    n = int(gib * 2 ** 30 // len(block))
    try:
        t0 = time.perf_counter()
        with open(fn, "wb", buffering=0) as f:
            for _ in range(n):
                f.write(block)
            os.fsync(f.fileno())
        out["write_mb_s"] = round(n * len(block) / (time.perf_counter() - t0) / 1e6)
        t0 = time.perf_counter()
        with open(fn, "rb", buffering=0) as f:
            while f.read(16 * 2 ** 20):
                pass
        out["read_mb_s"] = round(n * len(block) / (time.perf_counter() - t0) / 1e6)
        out["read_note"] = "may be served from page cache"
    finally:
        try:
            os.remove(fn)
        except OSError:
            pass
    return out


def download(seconds: float = 20.0, url: str = DOWNLOAD_URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "tokenwatt-check/0.1"})
    got, t0 = 0, time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            while time.perf_counter() - t0 < seconds:
                chunk = r.read(4 * 2 ** 20)
                if not chunk:
                    break
                got += len(chunk)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}
    el = time.perf_counter() - t0
    mbit = got * 8 / el / 1e6 if el else 0
    return {"mbit_s": round(mbit), "mb": round(got / 1e6), "seconds": round(el, 1), "source": "huggingface.co",
            "minutes_per_100gb": round(100e3 * 8 / mbit / 60, 1) if mbit else None}
