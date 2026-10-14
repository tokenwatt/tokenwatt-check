from tokenwatt_check.watch import Watcher


def row(**kw):
    r = {"index": 0, "name": "NVIDIA H100 80GB HBM3", "power.limit": 700.0, "power.draw": 650.0, "clocks.sm": 1400,
         "clocks.max.sm": 1980, "temperature.gpu": 60, "utilization.gpu": 100, "clocks_event_reasons.active": "0x4",
         "pcie.link.gen.current": 5, "pcie.link.width.current": 16, "ecc.errors.uncorrected.volatile.total": 0}
    r.update(kw)
    return r


def kinds(w):
    return [(e["level"], e["kind"]) for e in w.events]


def test_healthy_run_has_no_events():
    w = Watcher(quiet=True)
    for _ in range(10):
        w.check([row()])
    assert w.events == []


def test_power_limit_cut_mid_run():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row()])
    w.check([row(**{"power.limit": 525.0})])
    w.check([row(**{"power.limit": 525.0})])          # reported once, not every sample
    assert kinds(w) == [("WARN", "power_limit")]
    assert "75%" in w.events[0]["message"]


def test_clock_collapse_needs_three_loaded_samples():
    w = Watcher(quiet=True)
    w.check([row()])
    for _ in range(2):
        w.check([row(**{"clocks.sm": 787})])
    assert w.events == []
    w.check([row(**{"clocks.sm": 787})])
    assert kinds(w) == [("FAIL", "clock")]
    w.check([row()])
    assert kinds(w)[-1] == ("INFO", "clock")


def test_idle_low_clock_is_not_a_collapse():
    w = Watcher(quiet=True)
    for _ in range(6):
        w.check([row(**{"clocks.sm": 345, "utilization.gpu": 0, "clocks_event_reasons.active": "0x1"})])
    assert w.events == []


def test_thermal_throttle_start_and_end():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"clocks_event_reasons.active": "0x44"})])     # sw power cap + hw thermal slowdown
    w.check([row()])
    assert kinds(w) == [("FAIL", "throttle"), ("INFO", "throttle")]


def test_pcie_drop_and_new_ecc_errors():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"pcie.link.width.current": 8, "ecc.errors.uncorrected.volatile.total": 2})])
    assert ("WARN", "pcie") in kinds(w) and ("FAIL", "ecc") in kinds(w)


def test_summary_overall():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"power.limit": 525.0})])
    s = w.summary(5)
    assert s["overall"] == "WARN"
    assert s["gpus"][0]["power_limit_min"] == 525.0


def test_idle_pcie_gen1_is_not_a_drop():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"utilization.gpu": 0, "pcie.link.gen.current": 1})])   # idle power saving
    w.check([row()])
    assert ("WARN", "pcie") not in kinds(w)


def test_pcie_gen_drop_under_load():
    w = Watcher(quiet=True)
    w.check([row(**{"utilization.gpu": 0, "pcie.link.gen.current": 1})])   # watch started while idle
    w.check([row()])                                                       # Gen5 under load
    w.check([row(**{"pcie.link.gen.current": 3})])
    w.check([row(**{"pcie.link.gen.current": 3})])
    assert kinds(w).count(("WARN", "pcie")) == 1



def test_brief_software_throttle_is_warn_and_end_reports_dip(monkeypatch):
    import tokenwatt_check.watch as wm
    clock = {"t": 1000.0}
    monkeypatch.setattr(wm.time, "time", lambda: clock["t"])
    w = Watcher(quiet=True)
    w.check([row(**{"clocks_event_reasons.active": "0x4"})])            # baseline (power cap only)
    clock["t"] += 60
    w.check([row(**{"clocks.sm": 1400})])
    clock["t"] += 60
    w.check([row(**{"clocks_event_reasons.active": "0x20", "clocks.sm": 1250, "temperature.gpu": 82})])  # sw thermal
    clock["t"] += 60
    w.check([row(**{"clocks.sm": 1400})])
    levels = [(e["level"], e["message"]) for e in w.events if e["kind"] == "throttle"]
    assert levels[0][0] == "WARN" and "82 °C" in levels[0][1]
    assert levels[1][0] == "INFO" and "-11%" in levels[1][1] and "60 s" in levels[1][1]


def test_sustained_software_throttle_escalates_once(monkeypatch):
    import tokenwatt_check.watch as wm
    clock = {"t": 0.0}
    monkeypatch.setattr(wm.time, "time", lambda: clock["t"])
    w = Watcher(quiet=True)
    w.check([row()])
    for _ in range(8):
        clock["t"] += 60
        w.check([row(**{"clocks_event_reasons.active": "0x20"})])
    fails = [e for e in w.events if e["level"] == "FAIL"]
    assert len(fails) == 1 and "sustained" in fails[0]["message"]


def test_hardware_slowdown_is_fail_immediately():
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"clocks_event_reasons.active": "0x48"})])           # hw_slowdown + hw_thermal_slowdown
    assert [e["level"] for e in w.events if e["kind"] == "throttle"] == ["FAIL", "FAIL"]
