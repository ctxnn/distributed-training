# train_fsdp_mnist.py
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def get_loader(batch_size=64):
    ds = datasets.MNIST('.', download=True, transform=transforms.ToTensor())
    sampler = DistributedSampler(ds)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loader, sampler

def train(rank, world_size):
    setup()
    loader, sampler = get_loader()

    model = SimpleNet().cuda()
    # Wrap model for FSDP sharding
    fsdp_model = FSDP(model, sharding_strategy=FSDP.ShardingStrategy.FULL_SHARD)

    opt = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    start = time.time()
    fsdp_model.train()
    for epoch in range(1):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            opt.zero_grad()
            output = fsdp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    if dist.get_rank() == 0:
        elapsed = time.time() - start
        print(f"[FSDP] Epoch loss: {total_loss / len(loader):.4f}, Time: {elapsed:.2f}s")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)