"""Short GPU load tests with PyTorch: dense BF16 matmul (compute) and device-to-device copy (memory bandwidth)."""

from __future__ import annotations

import time


def torch_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"PyTorch not importable ({type(e).__name__}); install it with `pip install torch`"
    import torch
    if not torch.cuda.is_available():
        return False, "PyTorch can't see a CUDA device"
    return True, f"torch {torch.__version__}, CUDA {torch.version.cuda}"


def matmul_tflops(seconds: float = 30.0, n: int = 8192, device: int = 0) -> dict:
    """Sustained dense BF16 GEMM throughput. Runs long enough for clocks and power to settle."""
    import torch
    torch.cuda.set_device(device)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):                       # warm up kernels and clocks
        a @ b
    torch.cuda.synchronize()
    flop = 2 * n ** 3
    iters, t0 = 0, time.perf_counter()
    t_start = time.time()
    while True:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        iters += 10
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    t_end = time.time()
    del a, b
    torch.cuda.empty_cache()
    return {"tflops": round(flop * iters / el / 1e12, 1), "n": n, "dtype": "bf16", "seconds": round(el, 1),
            "t_start": t_start, "t_end": t_end}


def copy_bandwidth(seconds: float = 15.0, gib: float = 4.0, device: int = 0) -> dict:
    """Device-to-device copy bandwidth (read + write counted), the bound for LLM decode."""
    import torch
    torch.cuda.set_device(device)
    free, _total = torch.cuda.mem_get_info()
    gib = min(gib, free / 2 ** 30 / 2.5)      # two buffers plus headroom
    n = int(gib * 2 ** 30) // 2
    src = torch.empty(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    iters, t0 = 0, time.perf_counter()
    t_start = time.time()
    while True:
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()
        iters += 5
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    t_end = time.time()
    moved = 2 * n * 2 * iters                  # bytes read + written
    del src, dst
    torch.cuda.empty_cache()
    return {"tb_s": round(moved / el / 1e12, 2), "buffer_gib": round(gib, 2), "seconds": round(el, 1),
            "t_start": t_start, "t_end": t_end}
