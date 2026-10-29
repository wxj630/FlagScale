#!/usr/bin/env python3
"""Check whether the Ascend NPU stack supports FP64 GEMM."""

import torch
import torch_npu  # noqa: F401


def main():
    torch.npu.set_device(0)
    print(f"device: {torch.npu.get_device_name(0)}")
    print(f"torch: {torch.__version__}  torch_npu: {torch_npu.__version__}")

    print("\n[1] allocate float64 tensor on NPU")
    try:
        x = torch.ones(4, 4, dtype=torch.float64, device="npu")
        print(f"OK: dtype={x.dtype} device={x.device}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    print("\n[2] float64 matmul on NPU")
    for n in (8, 256, 1024):
        try:
            a = torch.randn(n, n, dtype=torch.float64, device="npu")
            b = torch.randn(n, n, dtype=torch.float64, device="npu")
            c = torch.matmul(a, b)
            torch.npu.synchronize()
            print(f"n={n}: OK dtype={c.dtype} sample={c[0, 0].item():.6e}")
        except Exception as e:
            print(f"n={n}: FAILED: {type(e).__name__}: {e}")

    print("\n[3] float64 elementwise add on NPU (for reference)")
    try:
        a = torch.ones(8, 8, dtype=torch.float64, device="npu")
        c = a + a
        torch.npu.synchronize()
        print(f"OK: dtype={c.dtype} sample={c[0, 0].item()}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
