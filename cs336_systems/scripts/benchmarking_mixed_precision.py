import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        print(f"Input dtype: {x.dtype}")
        x = self.relu(self.fc1(x))
        print(f"After fc1 dtype: {x.dtype}")
        x = self.ln(x)
        print(f"After ln dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"After fc2 dtype: {x.dtype}")
        return x


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ToyModel(1000, 10000).to(device=device)
    input = torch.randn(16, 1000).to(device=device)
    target = torch.randint(0, 10000, (16,)).to(device=device)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(10):
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model.forward(input).reshape(-1, 10000)
            print(f"Logits dtype: {logits.dtype}")
            loss = nn.functional.cross_entropy(logits, target)
            print(f"Loss dtype: {loss.dtype}")
        loss.backward()
        print(f"Grad dtype: {model.fc1.weight.grad.dtype}") 
        optim.step()
        optim.zero_grad()