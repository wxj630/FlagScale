import os, sys, time
dev = int(os.environ["CUDA_VISIBLE_DEVICES"])
import torch
torch.cuda.set_device(0)
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
t0 = time.time()
y = torch.mm(x, x); torch.cuda.synchronize()
print(f"card {dev}: OK {time.time()-t0:.3f}s", flush=True)
