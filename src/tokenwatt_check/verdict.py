"""Turn measurements into findings: PASS / WARN / FAIL with a reason and what to do about it."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# "Healthy" reference for this tool's own load tests: sustained dense BF16 GEMM (n=8192), the SM
# clock during that GEMM, and device-to-device copy bandwidth counted as read+write. Entries marked
# "measured" come from cards TokenWatt ran this tool on; the rest are provisional (vendor specs and
# public measurements) until we measure one. Matched against nvidia-smi's name, most specific first.
#
# A dense GEMM pulls a healthy card to its power limit, so its clock sits well below clocks.max.sm
# (an H100 SXM at 700 W holds ~1,360 MHz of 1,980). The clock check compares against `gemm_mhz`.
REFERENCE = [
    ("H100 NVL", {"tflops": 550, "tb_s": 3.3}),
    ("H100 PCIe", {"tflops": 450, "tb_s": 1.7}),
    ("H100", {"tflops": 680, "tb_s": 3.0, "gemm_mhz": 1360, "measured": "RunPod Secure Cloud, 700 W, 2026-10-09"}),  # SXM
    ("H200", {"tflops": 660, "tb_s": 4.3, "gemm_mhz": 1460, "measured": "RunPod Secure Cloud, 700 W, 2026-10-09"}),
    ("B200", {"tflops": 1400, "tb_s": 6.5}),
    ("A100-SXM4-80GB", {"tflops": 270, "tb_s": 1.7}),
    ("A100 80GB PCIe", {"tflops": 250, "tb_s": 1.6}),
    ("A100", {"tflops": 250, "tb_s": 1.3}),          # 40 GB
    ("L40S", {"tflops": 250, "tb_s": 0.75}),
    ("RTX 6000 Ada", {"tflops": 250, "tb_s": 0.8}),
    ("RTX 5090", {"tflops": 200, "tb_s": 1.5}),
    ("RTX 4090", {"tflops": 150, "tb_s": 0.85}),
    ("A10", {"tflops": 100, "tb_s": 0.5}),
]


def reference_for(name: str | None) -> tuple[str, dict] | None:
    if not name:
        return None
    for key, ref in REFERENCE:
        if key.lower() in name.lower():
            return key, ref
    return None


@dataclass
class Finding:
    level: str          # PASS | WARN | FAIL | INFO
    check: str
    message: str
    advice: str = ""


def evaluate(inv: dict, load: dict, host_disk: dict, host_net: dict) -> tuple[str, list[Finding], dict]:
    f: list[Finding] = []
    metrics: dict = {}

    # power cap
    lim, mx = inv.get("power.limit"), inv.get("power.max_limit")
    if isinstance(lim, (int, float)) and isinstance(mx, (int, float)) and mx:
        ratio = lim / mx
        metrics["power_limit_pct"] = round(ratio * 100)
        if ratio < 0.95:
            f.append(Finding("WARN", "power limit",
                             f"GPU power limit is {lim:.0f} W of a possible {mx:.0f} W ({ratio:.0%}).",
                             "The host lowered it. Expect lower throughput under heavy load; compare price per token, not per hour."))
        else:
            f.append(Finding("PASS", "power limit", f"Full power limit: {lim:.0f} W."))

    gemm = load.get("matmul") or {}
    tel = gemm.get("telemetry") or {}
    # clocks under load: against a healthy card's clock in the same GEMM when we have one,
    # otherwise against clocks.max.sm with looser thresholds
    ref = reference_for(inv.get("name"))
    mxclk, clk = inv.get("clocks.max.sm"), tel.get("sm_mhz_mean")
    base = (ref[1].get("gemm_mhz") if ref else None) or None
    if isinstance(clk, (int, float)) and (base or (isinstance(mxclk, (int, float)) and mxclk)):
        r = clk / (base or mxclk)
        if not base:
            r = min(1.0, r / 0.7)            # a healthy card at its power limit runs ~70% of max under GEMM
        metrics["clock_under_load_pct"] = round(r * 100)
        msg = (f"SM clock during the matmul averaged {clk:.0f} MHz — {r:.0%} of a healthy {ref[0]} ({base} MHz)"
               if base else f"SM clock during the matmul averaged {clk:.0f} MHz of {mxclk:.0f} MHz max")
        if r < 0.6:
            f.append(Finding("FAIL", "clocks under load", msg + ".",
                             "Far below normal. Possible causes: aggressive power cap, poor cooling or power delivery, a degraded card. Ask for another machine."))
        elif r < 0.8:
            f.append(Finding("WARN", "clocks under load", msg + ".", "Lower than a healthy card under this load; usually a power or thermal limit."))
        else:
            f.append(Finding("PASS", "clocks under load", msg + "."))

    reasons = tel.get("throttle_reasons") or {}
    hard = {k: v for k, v in reasons.items() if k in ("hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown", "hw_power_brake_slowdown") and v >= 0.1}
    if hard:
        f.append(Finding("FAIL", "throttling", "Hardware or thermal slowdown active during load: " +
                         ", ".join(f"{k} ({v:.0%} of samples)" for k, v in hard.items()) + ".",
                         "This is not a setting the host chose — the card is protecting itself. Avoid this machine."))
    elif reasons.get("sw_power_cap", 0) >= 0.5:
        f.append(Finding("INFO", "throttling", f"Running at its power limit for {reasons['sw_power_cap']:.0%} of the load test (normal under full load)."))

    # compute and bandwidth vs reference
    for key, val, unit, label in (("tflops", gemm.get("tflops"), "TFLOPS", "BF16 matmul"),
                                  ("tb_s", (load.get("copy") or {}).get("tb_s"), "TB/s", "memory copy bandwidth")):
        if not isinstance(val, (int, float)):
            continue
        metrics[key] = val
        if not ref:
            f.append(Finding("INFO", label, f"{val} {unit} (no reference yet for this GPU)."))
            continue
        pct = val / ref[1][key]
        metrics[f"{key}_vs_reference_pct"] = round(pct * 100)
        msg = f"{val} {unit} — {pct:.0%} of the healthy {ref[0]} reference ({ref[1][key]} {unit})"
        level = "FAIL" if pct < 0.75 else "WARN" if pct < 0.9 else "PASS"
        why = ("Prompt processing (prefill) and large batches scale with compute." if key == "tflops"
               else "LLM token generation (decode) speed follows memory bandwidth closely.")
        f.append(Finding(level, label, msg + ".", "" if level == "PASS" else f"Below what this GPU model normally delivers. {why}"))
    if ref is None and inv.get("name"):
        metrics["reference"] = None

    # PCIe link
    gmax, wmax = inv.get("pcie.link.gen.max"), inv.get("pcie.link.width.max")
    gcur, wcur = tel.get("pcie_gen_under_load", inv.get("pcie.link.gen.current")), tel.get("pcie_width_under_load", inv.get("pcie.link.width.current"))
    if all(isinstance(x, (int, float)) for x in (gmax, wmax, gcur, wcur)):
        if gcur < gmax or wcur < wmax:
            f.append(Finding("WARN", "PCIe link", f"PCIe link is Gen{gcur} x{wcur}, below the card's Gen{gmax} x{wmax}.",
                             "Slows model loading and host↔GPU transfers; mostly harmless once the model is loaded."))
        else:
            f.append(Finding("PASS", "PCIe link", f"Gen{gcur} x{wcur}."))

    # memory errors
    ecc, pend = inv.get("ecc.errors.uncorrected.volatile.total"), inv.get("retired_pages.pending")
    if isinstance(ecc, (int, float)) and ecc > 0:
        f.append(Finding("FAIL", "memory errors", f"{ecc:.0f} uncorrected ECC errors since the driver loaded.", "Unreliable memory. Do not run production work here."))
    if isinstance(pend, str) and pend.lower() == "yes":
        f.append(Finding("WARN", "memory errors", "Retired memory pages are pending a reset.", "The card needs a reset to retire bad memory."))

    tmax = tel.get("temp_c_max")
    if isinstance(tmax, (int, float)) and tmax >= 87:
        f.append(Finding("WARN", "temperature", f"Reached {tmax:.0f} °C under load.", "Hot; sustained work may throttle."))

    # host
    w = host_disk.get("write_mb_s")
    if isinstance(w, (int, float)):
        metrics["disk_write_mb_s"] = w
        f.append(Finding("WARN" if w < 1000 else "PASS", "disk", f"Sequential write {w:,} MB/s.",
                         "Slow disk: large models will take minutes to load." if w < 1000 else ""))
    mb = host_net.get("mbit_s")
    if isinstance(mb, (int, float)):
        metrics["download_mbit_s"] = mb
        mins = host_net.get("minutes_per_100gb")
        # single-stream download; the same host measured 0.9 and 9.5 Gbit/s on different days, so only flag clearly slow links
        f.append(Finding("WARN" if mb < 500 else "PASS", "download", f"{mb:,} Mbit/s from Hugging Face, one stream" +
                         (f" (~{mins} min per 100 GB of weights)." if mins else "."),
                         "You pay for the minutes spent downloading weights." if mb < 500 else ""))

    order = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}
    worst = max((order[x.level] for x in f), default=0)
    overall = {3: "FAIL", 2: "WARN"}.get(worst, "PASS")
    return overall, f, metrics


def findings_json(findings: list[Finding]) -> list[dict]:
    return [asdict(x) for x in findings]
