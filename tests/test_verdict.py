from tokenwatt_check.demo import DEMOS
from tokenwatt_check.gpu import _num, decode_reasons, summarize
from tokenwatt_check.verdict import evaluate, reference_for


def run(name):
    d = DEMOS[name]
    return evaluate(d["inventory"], d["load"], d["disk"], d["download"])


def test_demo_verdicts():
    assert run("healthy-h100")[0] == "PASS"
    assert run("capped-h100")[0] == "WARN"
    overall, findings, metrics = run("throttled-h100")
    assert overall == "FAIL"
    assert metrics["clock_under_load_pct"] < 60
    assert any(f.check == "clocks under load" and f.level == "FAIL" for f in findings)


def test_reference_matching_prefers_specific_names():
    assert reference_for("NVIDIA H100 PCIe")[0] == "H100 PCIe"
    assert reference_for("NVIDIA H100 80GB HBM3")[0] == "H100"
    assert reference_for("NVIDIA A100-SXM4-80GB")[0] == "A100-SXM4-80GB"
    assert reference_for("Tesla T4") is None


def test_nvidia_smi_parsing():
    assert _num("700.00 W") == 700.0
    assert _num("81559 MiB") == 81559
    assert _num("[N/A]") is None
    assert decode_reasons("0x0000000000000004") == ["sw_power_cap"]
    assert decode_reasons("0x0000000000000048") == ["hw_slowdown", "hw_thermal_slowdown"]


def test_hardware_slowdown_fails():
    d = DEMOS["healthy-h100"]
    load = {"matmul": {**d["load"]["matmul"], "telemetry": {**d["load"]["matmul"]["telemetry"],
                                                              "throttle_reasons": {"hw_slowdown": 0.4}}}, "copy": d["load"]["copy"]}
    overall, findings, _ = evaluate(d["inventory"], load, d["disk"], d["download"])
    assert overall == "FAIL" and any(f.check == "throttling" for f in findings)


def test_summarize_windows_and_reasons():
    s = [{"t": i, "clocks.sm": 1900 + i, "power.draw": 600.0, "reasons": ["sw_power_cap"] if i % 2 else []} for i in range(10)]
    out = summarize(s, 2, 7)
    assert out["samples"] == 6 and out["throttle_reasons"]["sw_power_cap"] == 0.5


def test_dcgm_failure_is_reported():
    d = DEMOS["healthy-h100"]
    load = {**d["load"], "_dcgm": {"level": 1, "result": "fail", "failed_tests": ["memory"], "tests_seen": 5}}
    overall, findings, _ = evaluate(d["inventory"], load, d["disk"], d["download"])
    assert overall == "FAIL" and any(f.check == "DCGM diagnostics" for f in findings)


def test_share_payload_drops_identifiers(monkeypatch):
    import json, urllib.request
    from tokenwatt_check import cli
    sent = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"id": "x"}'

    def fake(req, timeout=0):
        sent["body"] = json.loads(req.data)
        return R()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    cli.share({"tool": "tokenwatt-check", "host": {"hostname": "secret-host", "cpus": 8},
               "inventory": {"name": "NVIDIA H100", "uuid": "GPU-123"}}, "runpod")
    b = sent["body"]
    assert "hostname" not in b["host"] and "uuid" not in b["inventory"] and b["provider_hint"] == "runpod"


def test_reference_matching_respects_digit_boundaries():
    assert reference_for("NVIDIA A40")[0] == "A40"
    assert reference_for("NVIDIA RTX A4000") is None
    assert reference_for("NVIDIA A100-SXM4-80GB")[0] == "A100-SXM4-80GB"
    assert reference_for("NVIDIA A10")[0] == "A10"
    assert reference_for("NVIDIA L40S")[0] == "L40S"
    assert reference_for("NVIDIA GeForce RTX 4090")[0] == "RTX 4090"


def test_power_limit_compares_with_default_not_board_max():
    inv = {"name": "NVIDIA GeForce RTX 4090", "power.limit": 450.0, "power.max_limit": 500.0, "power.default_limit": 450.0}
    overall, findings, metrics = evaluate(inv, {}, {}, {})
    pl = next(f for f in findings if f.check == "power limit")
    assert pl.level == "PASS" and "500" in pl.message and metrics["power_limit_pct"] == 100
    inv["power.limit"] = 315.0
    assert next(f for f in evaluate(inv, {}, {}, {})[1] if f.check == "power limit").level == "WARN"



def test_provisional_reference_caps_at_warn_and_spec_power_floor():
    # H100 NVL as rented: limit 310 W reported as the default, spec range 350-400 W; clocks and matmul far below
    inv = {"name": "NVIDIA H100 NVL", "power.limit": 310.0, "power.default_limit": 310.0, "power.max_limit": 400.0,
           "clocks.max.sm": 1785}
    load = {"matmul": {"tflops": 339.1, "telemetry": {"sm_mhz_mean": 678}}, "copy": {"tb_s": 3.55}}
    overall, findings, _ = evaluate(inv, load, {}, {})
    lv = {f.check: f.level for f in findings}
    assert lv["power limit"] == "WARN" and "350" in next(f.message for f in findings if f.check == "power limit")
    assert lv["clocks under load"] == "WARN" and lv["BF16 matmul"] == "WARN"
    assert "provisional" in next(f.message for f in findings if f.check == "BF16 matmul")
    assert overall == "WARN"


def test_measured_reference_still_fails_hard():
    inv = {"name": "NVIDIA H100 80GB HBM3", "power.limit": 700.0, "power.default_limit": 700.0, "power.max_limit": 700.0,
           "clocks.max.sm": 1980}
    load = {"matmul": {"tflops": 300.0, "telemetry": {"sm_mhz_mean": 700}}}
    overall, findings, _ = evaluate(inv, load, {}, {})
    assert overall == "FAIL"
