import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from argparse import ArgumentParser

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_data_loader(batch_size=64, distributed=False, rank=0, world_size=1):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('.', download=True, transform=transform)

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler), sampler
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None

def train(model, device, loader, optimizer, sampler=None):
    model.train()
    total_loss = 0
    for epoch in range(1):
        if sampler:
            sampler.set_epoch(epoch)
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

def train_cpu():
    print("Training on CPU")
    device = torch.device("cpu")
    model = SimpleNet().to(device)
    loader, _ = get_data_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start = time.time()
    loss = train(model, device, loader, optimizer)
    print(f"[CPU] Loss: {loss:.4f}, Time: {time.time() - start:.2f} seconds")

def train_gpu():
    if not torch.cuda.is_available():
        print("No GPU found.")
        return
    print("Training on Single GPU")
    device = torch.device("cuda:0")
    model = SimpleNet().to(device)
    loader, _ = get_data_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start = time.time()
    loss = train(model, device, loader, optimizer)
    print(f"[GPU] Loss: {loss:.4f}, Time: {time.time() - start:.2f} seconds")

# ------------------ DDP Functions ------------------

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = SimpleNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loader, sampler = get_data_loader(distributed=True, rank=rank, world_size=world_size)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    start = time.time()
    loss = train(ddp_model, rank, loader, optimizer, sampler)
    if rank == 0:
        print(f"[DDP] Loss: {loss:.4f}, Time: {time.time() - start:.2f} seconds")
    cleanup()

# ------------------ Main ------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "gpu", "ddp"], required=True)
    parser.add_argument("--nproc", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    if args.mode == "cpu":
        train_cpu()
    elif args.mode == "gpu":
        train_gpu()
    elif args.mode == "ddp":
        world_size = args.nproc
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()