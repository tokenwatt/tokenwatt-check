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
  ❌ clocks under load      SM clock under load averaged 787 MHz of 1980 MHz max (40%).
                            → Far below normal. Possible causes: aggressive power cap, poor cooling or
                              power delivery, a degraded card. Ask for another machine.
  ❌ BF16 matmul            262 TFLOPS — 40% of the healthy H100 reference (650 TFLOPS).
  ⚠️  memory copy bandwidth  2.2 TB/s — 81% of the healthy H100 reference (2.7 TB/s).
  ✅ PCIe link              Gen5 x16.
  ✅ disk                   Sequential write 1,616 MB/s.
  ✅ download               2,100 Mbit/s from Hugging Face (~6.3 min per 100 GB of weights).
```

(Example modeled on a real node; try `tokenwatt-check --demo throttled-h100`, `capped-h100` or `healthy-h100` without a GPU.)

## What it checks

| Check | How | Why it matters |
|---|---|---|
| Power limit | `nvidia-smi` power.limit vs power.max_limit | Hosts lower it to save power; throughput drops under load |
| Clocks under load | SM clock sampled every second during a 30 s BF16 matmul | The clearest sign of a capped, hot or degraded card |
| Throttle reasons | nvidia-smi clock event reasons during load | Hardware or thermal slowdown means the card is protecting itself |
| Compute | Sustained BF16 GEMM (8192²) in TFLOPS | Prompt processing and large batches scale with it |
| Memory bandwidth | Device-to-device copy in TB/s | LLM token generation follows it closely |
| PCIe link | Generation and width under load vs the card's maximum | Slow links slow model loading |
| Memory errors | Uncorrected ECC errors, pending retired pages | Unreliable memory |
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
```

## Reference values

Compute and bandwidth are compared with a "healthy card" reference per GPU model (`src/tokenwatt_check/verdict.py`).
They start from vendor specifications and public measurements and are being calibrated against nodes we measure;
please open an issue with your JSON if a healthy card scores low. The power-limit and clock checks don't depend on
those references.

## Privacy

Everything runs locally. Nothing is sent anywhere except the 20-second download test from huggingface.co.

## Supported

NVIDIA GPUs with `nvidia-smi` (H100, H200, B200, A100, L40S, RTX 4090/5090 and others). AMD Instinct support is planned.

## About

Built by [TokenWatt](https://www.tokenwatt.io), which publishes vendor-neutral measurements of LLM inference —
SLO goodput, tokens per joule and cost per million tokens — on NVIDIA and AMD. MIT licensed.
