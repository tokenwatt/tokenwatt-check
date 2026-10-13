from tokenwatt_check.push import Pusher
from tokenwatt_check.watch import Watcher


def row(**kw):
    r = {"index": 0, "name": "NVIDIA H100 80GB HBM3", "power.limit": 700.0, "power.draw": 650.0, "clocks.sm": 1400,
         "clocks.max.sm": 1980, "temperature.gpu": 60, "utilization.gpu": 100, "clocks_event_reasons.active": "0x4",
         "pcie.link.gen.current": 5, "pcie.link.width.current": 16, "ecc.errors.uncorrected.volatile.total": 0}
    r.update(kw)
    return r


def test_reports_window_stats_and_each_event_once():
    sent = []
    p = Pusher("k", every=0, node="n1", post=lambda url, key, body: sent.append(body))
    w = Watcher(quiet=True)
    for pl in (700.0, 700.0, 500.0):
        rows = [row(**{"power.limit": pl, "power.draw": pl - 50})]
        w.check(rows)
        p.add(rows)
    assert p.maybe_flush(w.events)
    g = sent[0]["gpus"][0]
    assert g["last"]["power.limit"] == 500.0 and g["window"]["samples"] == 3
    assert g["window"]["power.draw"] == {"min": 450.0, "max": 650.0, "mean": 583.3}
    assert [e["kind"] for e in sent[0]["events"]] == ["power_limit"] and sent[0]["node"] == "n1"
    p.add([row()])
    p.maybe_flush(w.events)
    assert sent[1]["events"] == []                      # each event is sent once


def test_failed_post_keeps_events_for_next_report():
    calls = {"n": 0, "body": None}

    def flaky(url, key, body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("network down")
        calls["body"] = body
    p = Pusher("k", every=0, post=flaky)
    w = Watcher(quiet=True)
    w.check([row()])
    w.check([row(**{"power.limit": 400.0})])
    p.add([row()])
    assert not p.maybe_flush(w.events)
    p.add([row()])
    assert p.maybe_flush(w.events) and len(calls["body"]["events"]) == 1


def test_not_due_and_hostname_not_sent_by_default():
    p = Pusher("k", every=300, post=lambda *a: None)
    p.add([row()])
    assert not p.maybe_flush([])
    assert p.payload([])["node"] is None
