"""Watch mode: keep checking the GPUs while your workload runs, and flag changes as they happen.

A one-time check can't see a host lowering the power limit halfway through a rental, a card starting to
throttle when the room heats up, or a PCIe link renegotiating down. Watch mode samples every GPU every
--interval seconds, compares with the state at start, prints an event the moment something changes,
appends every sample to a JSONL log and prints a summary when you stop it (Ctrl-C) or --hours runs out.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from . import gpu

FIELDS = ["index", "name", "power.limit", "power.draw", "clocks.sm", "clocks.max.sm", "temperature.gpu",
          "utilization.gpu", "clocks_event_reasons.active", "pcie.link.gen.current", "pcie.link.width.current",
          "ecc.errors.uncorrected.volatile.total"]
BAD_REASONS = {"hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown", "hw_power_brake_slowdown"}
# Graded by effect (24-hour soak, 2026-10-14: 21 one-minute thermal throttles on an RTX 4090 cost < 1% throughput):
# hardware slowdowns are the card protecting itself → FAIL at once; a software thermal slowdown is a WARN unless
# it lasts SUSTAINED_SECONDS, then it's escalated to FAIL. Every episode's end reports its duration and clock dip.
HARD_REASONS = {"hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
SUSTAINED_SECONDS = 300
CLOCK_COLLAPSE = 0.5      # SM clock under load below this share of max (healthy cards stay ~0.65+ even at the power limit)
COLLAPSE_SAMPLES = 3      # consecutive samples before flagging
LOAD_UTIL = 90            # utilization % that counts as "under load"
HOT_C = 87


def _now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _num(v):
    return v if isinstance(v, (int, float)) else None


class Watcher:
    def __init__(self, on_event: str | None = None, quiet: bool = False):
        self.on_event = on_event
        self.quiet = quiet
        self.base: dict[int, dict] = {}
        self.state: dict[int, dict] = {}
        self.stats: dict[int, dict] = {}
        self.events: list[dict] = []
        self.samples = 0

    def event(self, level: str, g: int, kind: str, msg: str) -> None:
        self.events.append({"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "gpu": g,
                            "level": level, "kind": kind, "message": msg})
        if not self.quiet:
            icon = {"FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️ "}[level]
            print(f"[{_now()}] {icon} GPU {g}: {msg}", flush=True)
        if self.on_event:
            env = dict(os.environ, TWC_LEVEL=level, TWC_GPU=str(g), TWC_KIND=kind, TWC_MESSAGE=msg)
            try:
                subprocess.run(self.on_event, shell=True, timeout=30, env=env)  # noqa: S602 — user-supplied hook
            except Exception as ex:  # noqa: BLE001
                print(f"  (--on-event command failed: {type(ex).__name__})", flush=True)

    def check(self, rows: list[dict]) -> None:
        self.samples += 1
        for r in rows:
            g = r.get("index")
            if not isinstance(g, int):
                continue
            reasons = set(gpu.decode_reasons(r.get("clocks_event_reasons.active")))
            if g not in self.base:
                self.base[g] = dict(r)
                self.state[g] = {"reasons": reasons, "collapse": 0, "collapsed": False, "hot": False,
                                 "pl": _num(r.get("power.limit")), "ecc": _num(r.get("ecc.errors.uncorrected.volatile.total"))}
                self.stats[g] = {"min_power_limit": _num(r.get("power.limit")), "loaded": 0, "collapsed": 0,
                                 "reason_counts": {}, "n": 0}
                continue
            b, s, st = self.base[g], self.state[g], self.stats[g]
            st["n"] += 1

            pl = _num(r.get("power.limit"))
            if pl is not None and s["pl"] is not None and abs(pl - s["pl"]) / s["pl"] > 0.02:
                start = _num(b.get("power.limit")) or s["pl"]
                drop = pl < s["pl"]
                self.event("WARN" if drop else "INFO", g, "power_limit",
                           f"power limit changed from {s['pl']:.0f} W to {pl:.0f} W ({pl / start:.0%} of the {start:.0f} W at start)"
                           + (" — throughput will drop" if drop else ""))
                s["pl"] = pl
            if pl is not None:
                st["min_power_limit"] = min(st["min_power_limit"] if st["min_power_limit"] is not None else pl, pl)

            now = time.time()
            sm_now, t_now = _num(r.get("clocks.sm")), _num(r.get("temperature.gpu"))
            eps = s.setdefault("episodes", {})
            for name in sorted((reasons & BAD_REASONS) - s["reasons"]):
                hard = name in HARD_REASONS
                eps[name] = {"start": now, "clock_before": s.get("last_clock"), "min_clock": sm_now, "max_temp": t_now,
                             "escalated": False}
                self.event("FAIL" if hard else "WARN", g, "throttle",
                           f"{name.replace('_', ' ')} started"
                           + (f" at {t_now:.0f} °C" if t_now is not None else "")
                           + (" (the card protecting itself)" if hard else ""))
            for name in reasons & BAD_REASONS & s["reasons"]:
                ep = eps.get(name)
                if not ep:
                    continue
                if sm_now is not None:
                    ep["min_clock"] = sm_now if ep["min_clock"] is None else min(ep["min_clock"], sm_now)
                if t_now is not None:
                    ep["max_temp"] = t_now if ep["max_temp"] is None else max(ep["max_temp"], t_now)
                if name not in HARD_REASONS and not ep["escalated"] and now - ep["start"] >= SUSTAINED_SECONDS:
                    ep["escalated"] = True
                    self.event("FAIL", g, "throttle", f"{name.replace('_', ' ')} sustained for "
                               f"{(now - ep['start']) / 60:.0f} min — throughput is likely affected")
            for name in sorted((s["reasons"] & BAD_REASONS) - reasons):
                ep = eps.pop(name, None) or {}
                dur = now - ep["start"] if ep.get("start") else None
                dip = ""
                if ep.get("min_clock") and ep.get("clock_before"):
                    dip = f"; lowest SM clock {ep['min_clock']:.0f} MHz ({ep['min_clock'] / ep['clock_before'] - 1:+.0%})"
                self.event("INFO", g, "throttle", f"{name.replace('_', ' ')} ended"
                           + (f" after {dur / 60:.0f} min" if dur and dur >= 90 else f" after {dur:.0f} s" if dur else "")
                           + dip + (f"; peak {ep['max_temp']:.0f} °C" if ep.get("max_temp") is not None else ""))
            if sm_now is not None and not (reasons & BAD_REASONS):
                s["last_clock"] = sm_now
            s["reasons"] = reasons
            for name in reasons:
                st["reason_counts"][name] = st["reason_counts"].get(name, 0) + 1

            util, sm, smax = _num(r.get("utilization.gpu")), _num(r.get("clocks.sm")), _num(r.get("clocks.max.sm"))
            loaded = util is not None and util >= LOAD_UTIL
            if loaded:
                st["loaded"] += 1
                if sm and smax and sm < CLOCK_COLLAPSE * smax:
                    s["collapse"] += 1
                    st["collapsed"] += 1
                    if s["collapse"] == COLLAPSE_SAMPLES and not s["collapsed"]:
                        s["collapsed"] = True
                        self.event("FAIL", g, "clock",
                                   f"SM clock under load is {sm} MHz, {sm / smax:.0%} of max ({smax} MHz) — far below a "
                                   "healthy card; your work is running much slower")
                else:
                    if s["collapsed"]:
                        self.event("INFO", g, "clock", f"SM clock under load recovered to {sm} MHz")
                    s["collapse"], s["collapsed"] = 0, False

            t = _num(r.get("temperature.gpu"))
            if t is not None and t >= HOT_C and not s["hot"]:
                s["hot"] = True
                self.event("WARN", g, "temperature", f"temperature {t} °C")
            elif t is not None and t < HOT_C - 5:
                s["hot"] = False

            w, bw = _num(r.get("pcie.link.width.current")), _num(b.get("pcie.link.width.current"))
            if w and bw and w < bw and w != s.get("pcie.link.width.current"):
                self.event("WARN", g, "pcie", f"PCIe link width dropped from {bw} to {w}")
            s["pcie.link.width.current"] = w
            # Idle GPUs drop the link to Gen1 to save power and go back up under load, so the generation only
            # counts under load, against the highest generation this GPU has shown.
            gen = _num(r.get("pcie.link.gen.current"))
            peak = max(x for x in (s.get("gen_peak"), _num(b.get("pcie.link.gen.current")), gen) if x is not None) \
                if gen is not None else s.get("gen_peak")
            if loaded and gen and peak and gen < peak and gen != s.get("gen_reported"):
                self.event("WARN", g, "pcie", f"PCIe generation under load dropped from {peak} to {gen}")
                s["gen_reported"] = gen
            elif loaded and gen == peak:
                s["gen_reported"] = None
            s["gen_peak"] = peak

            ecc = _num(r.get("ecc.errors.uncorrected.volatile.total"))
            if ecc is not None and s["ecc"] is not None and ecc > s["ecc"]:
                self.event("FAIL", g, "ecc", f"uncorrected ECC errors rose from {s['ecc']} to {ecc}")
                s["ecc"] = ecc

    def summary(self, minutes: float) -> dict:
        out = {"minutes": round(minutes, 1), "samples": self.samples, "events": self.events, "gpus": {}}
        print(f"\nwatched {len(self.base)} GPU(s) for {minutes:.0f} min · {self.samples} samples · {len(self.events)} event(s)")
        for g, st in self.stats.items():
            b, n = self.base[g], max(st["n"], 1)
            reasons = {k: round(v / n, 2) for k, v in sorted(st["reason_counts"].items())}
            out["gpus"][g] = {"name": b.get("name"), "power_limit_start": b.get("power.limit"),
                              "power_limit_min": st["min_power_limit"], "loaded_share": round(st["loaded"] / n, 2),
                              "collapsed_samples": st["collapsed"], "reason_share": reasons}
            print(f"  GPU {g} {b.get('name')}: power limit {b.get('power.limit')} W at start, lowest {st['min_power_limit']} W · "
                  f"under load {st['loaded'] / n:.0%} of samples · clock collapsed in {st['collapsed']} · "
                  f"clock event reasons {reasons or 'none'}")
        overall = ("FAIL" if any(e["level"] == "FAIL" for e in self.events) else
                   "WARN" if any(e["level"] == "WARN" for e in self.events) else "PASS")
        out["overall"] = overall
        print(f"  OVERALL: {overall}")
        return out


def run(interval: float = 60, hours: float | None = None, log: str | None = None, on_event: str | None = None,
        pusher=None) -> int:
    w = Watcher(on_event)
    path = Path(log or f"tokenwatt-watch-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl")
    started = time.time()
    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))
    print(f"watching every {interval:.0f} s{f' for {hours:g} h' if hours else ''} · log {path} · Ctrl-C to stop and summarize"
          + (f" · reporting to {pusher.url} every {pusher.every:.0f} s" if pusher else ""), flush=True)
    with path.open("a") as f:
        while not stop["now"] and not (hours and time.time() - started > hours * 3600):
            t = time.time()
            try:
                rows = gpu.query(FIELDS)
            except Exception as ex:  # noqa: BLE001
                print(f"[{_now()}] sample failed: {type(ex).__name__}", flush=True)
                rows = []
            if rows:
                first = w.samples == 0
                w.check(rows)
                if pusher:
                    pusher.add(rows)
                    pusher.maybe_flush(w.events)
                f.write(json.dumps({"t": round(t, 1), "gpus": rows}, default=str) + "\n")
                f.flush()
                if first:
                    for r in rows:
                        print(f"  GPU {r.get('index')} {r.get('name')}: power limit {r.get('power.limit')} W, max clock "
                              f"{r.get('clocks.max.sm')} MHz, PCIe Gen{r.get('pcie.link.gen.current')} "
                              f"x{r.get('pcie.link.width.current')}", flush=True)
            while not stop["now"] and time.time() - t < interval:
                time.sleep(min(1.0, interval))
    summary = w.summary((time.time() - started) / 60)
    if pusher:
        pusher.maybe_flush(w.events, final=True)
    with path.open("a") as f:
        f.write(json.dumps({"summary": summary}, default=str) + "\n")
    return {"PASS": 0, "WARN": 0, "FAIL": 2}[summary["overall"]]
