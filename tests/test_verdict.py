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
