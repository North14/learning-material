import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, functional


# ---------------------------
# 1. Simple Spiking CNN Model
# ---------------------------
class SimpleSNN(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T  # simulation steps

        # Standard conv layer
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # Spiking neuron: Leaky Integrate-and-Fire
        self.lif = neuron.LIFNode(tau=2.0)

        # Classifier
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        spk_out = 0

        # Run for T timesteps
        for _ in range(self.T):
            out = self.conv(x)
            out = self.lif(out)               # apply spike activation
            spk_out += out                    # accumulate spikes

        functional.reset_net(self)            # reset membrane after forward

        spk_out = spk_out.flatten(1)
        return self.fc(spk_out)


# ---------------------------
# 2. Training Loop
# ---------------------------
def train_snn():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Model
    model = SimpleSNN(T=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    for epoch in range(1, 3):
        model.train()
        total = 0
        correct = 0

        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            # Accuracy
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")


if __name__ == "__main__":
    train_snn()
