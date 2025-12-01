import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
from max_former import Max_Former   # <-- your implementation

from torchvision.datasets import FakeData

# ------------------------------------------
# 1. Create a super-minimal dataset loader
# ------------------------------------------
def get_dataloader(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)


# ------------------------------------------
# 2. Use YOUR MaxFormer model
# ------------------------------------------
def create_model():
    model = Max_Former(
        in_channels=3,
        num_classes=10,
        embed_dims=384,
        mlp_ratios=4,
        depths=[1, 1, 2],     # tiny config
        T=4                   # simulation steps
    )
    return model


# ------------------------------------------
# 3. Tiny training loop
# ------------------------------------------
def train_one_epoch(model, loader, device):
    model.to(device)
    model.train()

    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for step, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)

        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        opt.step()

        functional.reset_net(model)

        print(f"[step {step}] loss = {loss.item():.4f}")

        if step == 100:
            break  # keep it ultra short for testing


# ------------------------------------------
# 4. Main
# ------------------------------------------
if __name__ == "__main__":
    device = "cpu"   # force CPU (your PyTorch does NOT support GTX 1080)

    print("Loading data...")
    loader = get_dataloader()

    print("Creating model...")
    model = create_model()

    print("Training...")
    train_one_epoch(model, loader, device)

    print("Done.")
