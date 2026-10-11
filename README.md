# tokenwatt-check

**Is the GPU you rented performing like it should?**

GPU listings tell you the model — "H100 SXM, verified, 99.5% reliability" — not the card you get. When we
[benchmarked four rented H100s](https://www.tokenwatt.io/blog/runpod-vs-vast-vs-lambda-h100.html), two had their
power limit lowered by the host and one ran at **787 MHz instead of ~1,900** under load: every token took 2.9× longer.
Nothing on the listing showed it.

`tokenwatt-check` finds that in about two minutes:

```
pip install tokenwatt-check torch    # torch is usually already there on GPU images
tokenwatt-check
```

```
  OVERALL: ❌ FAIL

  ⚠️  power limit            GPU power limit is 500 W of a possible 700 W (71%).
  ❌ clocks under load      SM clock during the matmul averaged 560 MHz — 41% of a healthy H100 (1360 MHz).
                            → Far below normal. Possible causes: aggressive power cap, poor cooling or
                              power delivery, a degraded card. Ask for another machine.
  ❌ BF16 matmul            280 TFLOPS — 41% of the healthy H100 reference (680 TFLOPS).
  ❌ memory copy bandwidth  2.2 TB/s — 73% of the healthy H100 reference (3.0 TB/s).
  ✅ PCIe link              Gen5 x16.
  ✅ disk                   Sequential write 1,616 MB/s.
  ✅ download               2,100 Mbit/s from Hugging Face, one stream (~6.3 min per 100 GB of weights).
```

(Illustrative output modeled on a real under-performing node — see `src/tokenwatt_check/demo.py`. Try `tokenwatt-check --demo throttled-h100`, `capped-h100` or `healthy-h100` without a GPU.)

## What it checks

| Check | How | Why it matters |
|---|---|---|
| Power limit | `nvidia-smi` power.limit vs power.max_limit | Hosts lower it to save power; throughput drops under load |
| Clocks under load | SM clock sampled every second during a 30 s BF16 matmul, compared with a healthy card of the same model in the same test | The clearest sign of a capped, hot or degraded card |
| Throttle reasons | nvidia-smi clock event reasons during load | Hardware or thermal slowdown means the card is protecting itself |
| Compute | Sustained BF16 GEMM (8192²) in TFLOPS | Prompt processing and large batches scale with it |
| Memory bandwidth | Device-to-device copy in TB/s | LLM token generation follows it closely |
| PCIe link | Generation and width under load vs the card's maximum | Slow links slow model loading |
| Memory errors | Uncorrected ECC errors, pending retired pages | Unreliable memory |
| DCGM diagnostics | `dcgmi diag -r 1` when NVIDIA DCGM is installed (`--dcgm 2` for the longer run) | NVIDIA's own hardware checks |
| Disk | 2 GB sequential write and read | How long weights take to load from disk |
| Download | 20 s from the Hugging Face CDN | You pay for the minutes spent downloading weights |

Results: `PASS`, `WARN` or `FAIL` per check with a reason and what to do, plus a JSON file with every reading
(`--json path`) and an optional Markdown report (`--md path`). Exit code 2 on `FAIL`, for scripts.

## Options

```
tokenwatt-check --gpu 1            # another GPU
tokenwatt-check --seconds 60       # longer load test
tokenwatt-check --no-load          # inventory, disk and network only (no PyTorch needed)
tokenwatt-check --no-net --no-disk
tokenwatt-check --share --provider runpod   # add an anonymized copy to the public node statistics
tokenwatt-check --watch            # keep watching during your workload (see below)
```

## Watch mode: catch changes during the rental

A check at the start can't see a host lowering the power limit halfway through your rental, a card that starts
throttling when the room heats up, or a PCIe link that renegotiates down. Leave watch mode running next to your workload:

```
tokenwatt-check --watch                       # every 60 s, all GPUs, until Ctrl-C
tokenwatt-check --watch --interval 30 --hours 24 --log node-a.jsonl
tokenwatt-check --watch --on-event 'curl -s -X POST -d "$TWC_MESSAGE" https://hooks.example.com/gpu'
```

```
watching every 60 s · log tokenwatt-watch-20261010-141500.jsonl · Ctrl-C to stop and summarize
  GPU 0 NVIDIA H100 80GB HBM3: power limit 700.0 W, max clock 1980 MHz, PCIe Gen5 x16
[16:42:07] ⚠️  GPU 0: power limit changed from 700 W to 525 W (75% of the 700 W at start) — throughput will drop
[16:44:07] ❌ GPU 0: SM clock under load is 787 MHz, 40% of max (1980 MHz) — far below a healthy card; your work is running much slower
```

It flags, the moment they happen: power-limit changes, hardware or thermal throttling starting and ending, the SM
clock under load falling below half of max for three samples in a row (healthy cards stay around two-thirds even at
their power limit), temperature at 87 °C or more, PCIe width or generation dropping, and new uncorrected ECC errors.
Every sample goes to a JSONL log; when you stop it you get a summary per GPU (lowest power limit, share of time under
load and throttled) and an overall PASS / WARN / FAIL. Watch mode only reads `nvidia-smi`, so it adds no load.

## Reference values

Compute, bandwidth and clock-under-load are compared with a healthy card of the same model (`src/tokenwatt_check/verdict.py`):

| GPU | BF16 matmul | Clock during matmul | Copy bandwidth | Source |
|---|---|---|---|---|
| H100 SXM | 680 TFLOPS | 1,360 MHz | 3.0 TB/s | measured, RunPod Secure Cloud, 700 W |
| H200 | 660 TFLOPS | 1,460 MHz | 4.3 TB/s | measured, RunPod Secure Cloud, 700 W |
| A100 SXM 80GB | 260 TFLOPS | 1,235 MHz | 1.75 TB/s | measured, 3 nodes, 400 W |
| L40S | 170 TFLOPS | 1,215 MHz | 0.65 TB/s | measured, 2 nodes, 350 W |
| RTX 4090 | 160 TFLOPS | 2,480 MHz | 0.92 TB/s | measured, 3 nodes, 450 W |
| A40 | 107 TFLOPS | 1,300 MHz | 0.56 TB/s | measured, 2 nodes, 300 W |
| others | | | | provisional (vendor specs, public measurements) |

Copy bandwidth is what this test reaches, not the spec sheet: GDDR6 cards (L40S, A40, RTX 4090) reach roughly
65–75% of their rated bandwidth in a device-to-device copy, HBM cards about 90%.

The power-limit check compares with the card's default limit, not the board maximum: an RTX 4090 at its stock
450 W is full power even if its board allows 500 W.

A dense matmul pulls a healthy card to its power limit, so its clock sits well below the maximum — a healthy H100 holds
about 69% of max clock in this test. That's why the clock check compares against a healthy card rather than the maximum.
If a healthy card scores low on your machine, please open an issue with your JSON so we can improve the references.

## Privacy

Everything runs locally. Nothing is sent anywhere except the 20-second download test from huggingface.co —
unless you add `--share`. Then one anonymized report goes to TokenWatt: GPU model, driver, limits and measurements,
CPU count and RAM size, and the provider name if you give one. Hostnames, GPU UUIDs, paths and your IP address are
not kept. Shared reports are published only as aggregate statistics per GPU model, which is how the healthy-card
references get better for everyone. Details: [privacy policy](https://www.tokenwatt.io/privacy.html)

## Supported

NVIDIA GPUs with `nvidia-smi` (H100, H200, B200, A100, L40S, RTX 4090/5090 and others). AMD Instinct support is planned.

## About

Built by [TokenWatt](https://www.tokenwatt.io), which publishes vendor-neutral measurements of LLM inference —
SLO goodput, tokens per joule and cost per million tokens — on NVIDIA and AMD. MIT licensed.
