"""GPU inventory and live sampling through nvidia-smi (no Python bindings needed)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

STATIC_FIELDS = [
    "index", "name", "uuid", "driver_version", "memory.total", "pcie.link.gen.max", "pcie.link.width.max",
    "pcie.link.gen.current", "pcie.link.width.current", "power.limit", "power.max_limit", "power.default_limit",
    "clocks.max.sm", "clocks.max.memory", "ecc.mode.current", "ecc.errors.uncorrected.volatile.total",
    "retired_pages.pending", "temperature.gpu", "persistence_mode", "compute_mode",
]
SAMPLE_FIELDS = ["index", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu",
                 "pcie.link.gen.current", "pcie.link.width.current", "clocks_event_reasons.active"]

# nvidia-smi clocks_event_reasons bitmask
REASONS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _num(v: str):
    v = v.strip()
    if v in ("[N/A]", "N/A", "[Not Supported]", "Not Supported", ""):
        return None
    for unit in (" MiB", " W", " MHz", " %"):
        if v.endswith(unit):
            v = v[: -len(unit)]
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return v


def query(fields: list[str]) -> list[dict]:
    """Run nvidia-smi --query-gpu for these fields; unsupported fields come back as None."""
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        # some drivers reject unknown fields as a whole; retry field by field
        rows: list[dict] = []
        for f in fields:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            vals = r.stdout.strip().splitlines() if r.returncode == 0 else []
            for i, v in enumerate(vals):
                while len(rows) <= i:
                    rows.append({})
                rows[i][f] = _num(v)
        return rows
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        rows.append({f: _num(p) for f, p in zip(fields, parts)})
    return rows


def inventory() -> list[dict]:
    return query(STATIC_FIELDS)


def decode_reasons(mask) -> list[str]:
    if isinstance(mask, str):
        try:
            mask = int(mask, 16)
        except ValueError:
            return []
    if not isinstance(mask, int):
        return []
    return [name for bit, name in REASONS.items() if mask & bit]


@dataclass
class Sampler:
    """Samples clocks, power, temperature and throttle reasons once a second in a thread."""
    gpu: int = 0
    period: float = 1.0
    samples: list[dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            try:
                rows = query(SAMPLE_FIELDS)
                if self.gpu < len(rows):
                    s = rows[self.gpu]
                    s["t"] = t
                    s["reasons"] = decode_reasons(s.get("clocks_event_reasons.active"))
                    self.samples.append(s)
            except Exception:  # noqa: BLE001 — keep sampling
                pass
            self._stop.wait(max(0.0, self.period - (time.time() - t)))

    def start(self) -> "Sampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self.samples


def summarize(samples: list[dict], t0: float, t1: float) -> dict:
    s = [x for x in samples if t0 <= x["t"] <= t1]
    if not s:
        return {}

    def vals(k):
        return [x[k] for x in s if isinstance(x.get(k), (int, float))]
    out = {"samples": len(s)}
    for k, name in (("clocks.sm", "sm_mhz"), ("clocks.mem", "mem_mhz"), ("power.draw", "power_w"),
                    ("temperature.gpu", "temp_c"), ("utilization.gpu", "util_pct")):
        v = vals(k)
        if v:
            out[f"{name}_mean"] = round(sum(v) / len(v), 1)
            out[f"{name}_min"] = min(v)
            out[f"{name}_max"] = max(v)
    reasons: dict[str, int] = {}
    for x in s:
        for r in x.get("reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    out["throttle_reasons"] = {k: round(v / len(s), 2) for k, v in reasons.items()}   # share of samples
    gens = vals("pcie.link.gen.current")
    widths = vals("pcie.link.width.current")
    if gens:
        out["pcie_gen_under_load"] = max(gens)
    if widths:
        out["pcie_width_under_load"] = max(widths)
    return out
