"""Report watch-mode results to a monitoring endpoint (TokenWatt Monitor, or your own).

Every `every` seconds the watcher posts one small JSON document: per GPU, the latest reading and the
min / mean / max of power, SM clock, temperature and utilization since the last report, plus any events
(power-limit cuts, throttling, clock collapse, PCIe or ECC changes) not sent yet. Only GPU telemetry is
sent: no hostnames unless you name the node, no processes, files or environment.

The key comes from TOKENWATT_KEY or a file (--key-file), never from the command line, where other users
on the machine could see it in `ps`. A failed post is retried with the next report; events are kept until
they're delivered.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://www.tokenwatt.io/api/monitor"
STAT_FIELDS = ("power.draw", "clocks.sm", "temperature.gpu", "utilization.gpu")
KEEP = ("index", "name", "power.limit", "power.draw", "clocks.sm", "clocks.max.sm", "temperature.gpu", "utilization.gpu",
        "pcie.link.gen.current", "pcie.link.width.current", "ecc.errors.uncorrected.volatile.total")


def load_key(key_file: str | None) -> str | None:
    if key_file:
        return Path(key_file).expanduser().read_text().strip() or None
    return os.environ.get("TOKENWATT_KEY") or None


class Pusher:
    def __init__(self, key: str, url: str = DEFAULT_URL, every: float = 300, node: str | None = None,
                 version: str = "", post=None):
        self.key, self.url, self.every, self.node, self.version = key, url, every, node, version
        # stable, anonymous id so several nodes on one account stay apart; the hostname itself isn't sent
        self.node_id = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
        self.post = post or self._post            # injectable for tests
        self.last = time.time()
        self.sent_events = 0
        self.failures = 0
        self._reset()

    def _reset(self) -> None:
        self.window_start = time.time()
        self.stats: dict[int, dict] = {}
        self.latest: dict[int, dict] = {}

    def add(self, rows: list[dict]) -> None:
        for r in rows:
            g = r.get("index")
            self.latest[g] = {k: r.get(k) for k in KEEP}
            st = self.stats.setdefault(g, {"n": 0, **{f: {"min": None, "max": None, "sum": 0.0, "n": 0} for f in STAT_FIELDS}})
            st["n"] += 1
            for f in STAT_FIELDS:
                v = r.get(f)
                if isinstance(v, (int, float)):
                    s = st[f]
                    s["min"] = v if s["min"] is None else min(s["min"], v)
                    s["max"] = v if s["max"] is None else max(s["max"], v)
                    s["sum"] += v
                    s["n"] += 1

    def payload(self, events: list[dict], final: bool = False) -> dict:
        gpus = []
        for g, last in sorted(self.latest.items(), key=lambda kv: (kv[0] is None, kv[0])):
            st = self.stats.get(g, {})
            agg = {f: {"min": s["min"], "max": s["max"], "mean": round(s["sum"] / s["n"], 1) if s["n"] else None}
                   for f, s in st.items() if f != "n"}
            gpus.append({"index": g, "last": last, "window": {"samples": st.get("n", 0), **agg}})
        return {"schema": 1, "agent": f"tokenwatt-check {self.version}".strip(), "node": self.node, "node_id": self.node_id,
                "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "window_s": round(time.time() - self.window_start),
                "final": final, "gpus": gpus, "events": events[self.sent_events:]}

    def maybe_flush(self, events: list[dict], final: bool = False) -> bool:
        if not final and time.time() - self.last < self.every:
            return False
        self.last = time.time()
        if not self.latest:
            return False
        body = self.payload(events, final)
        try:
            self.post(self.url, self.key, body)
        except Exception as ex:  # noqa: BLE001 — monitoring must never stop the watch
            self.failures += 1
            if self.failures in (1, 10) or self.failures % 100 == 0:
                print(f"  (report to {self.url} failed: {type(ex).__name__}; will retry with the next report)", flush=True)
            return False
        if self.failures:
            print(f"  (reporting resumed after {self.failures} failed attempt(s))", flush=True)
        self.failures = 0
        self.sent_events = len(events)
        self._reset()
        return True

    @staticmethod
    def _post(url: str, key: str, body: dict) -> None:
        req = urllib.request.Request(url, data=json.dumps(body, default=str).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": body.get("agent") or "tokenwatt-check"})
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status >= 300:
                raise RuntimeError(f"HTTP {r.status}")
