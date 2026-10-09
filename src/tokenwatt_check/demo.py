"""Example inputs for --demo, modeled on three H100 SXM nodes TokenWatt rented in October 2026.

The healthy example is a real tokenwatt-check run (RunPod Secure Cloud H100, 2026-10-09). The capped
and throttled examples use the power limits, inference-load clocks and host speeds we read on two
Vast.ai hosts; their matmul clocks and throughput are estimated from those readings, because those
runs predate this tool's load test.
"""


def _h100(name, power_limit, clk, tflops, tb_s, disk_w, mbit, reasons, note):
    return {
        "note": note,
        "host": {"hostname": name},
        "inventory": {"name": "NVIDIA H100 80GB HBM3", "driver_version": "580.126.09", "memory.total": 81559,
                      "pcie.link.gen.max": 5, "pcie.link.width.max": 16, "pcie.link.gen.current": 5,
                      "pcie.link.width.current": 16, "power.limit": power_limit, "power.max_limit": 700,
                      "clocks.max.sm": 1980, "ecc.errors.uncorrected.volatile.total": 0, "retired_pages.pending": "No"},
        "load": {"matmul": {"tflops": tflops, "telemetry": {"sm_mhz_mean": clk, "temp_c_max": 58,
                                                            "pcie_gen_under_load": 5, "pcie_width_under_load": 16,
                                                            "throttle_reasons": reasons}},
                 "copy": {"tb_s": tb_s}},
        "disk": {"write_mb_s": disk_w},
        "download": {"mbit_s": mbit, "minutes_per_100gb": round(100e3 * 8 / mbit / 60, 1)},
    }


DEMOS = {
    "healthy-h100": _h100("demo-healthy", 700, 1359, 682.3, 3.05, 1726, 9400, {"sw_power_cap": 1.0},
                          "healthy H100 SXM (modeled on a RunPod Secure Cloud node)"),
    "capped-h100": _h100("demo-capped", 525, 1150, 575, 2.9, 4400, 6000, {"sw_power_cap": 1.0},
                         "power-capped H100 SXM (modeled on a Vast.ai datacenter host capped at 525 W)"),
    "throttled-h100": _h100("demo-throttled", 500, 560, 280, 2.2, 1616, 2100, {"sw_power_cap": 1.0},
                            "under-performing H100 SXM (modeled on a Vast.ai community host: 787 MHz at a 500 W cap)"),
}
