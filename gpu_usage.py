import torch

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: {free / 1024**2:.2f} MiB free / {total / 1024**2:.2f} MiB total")
