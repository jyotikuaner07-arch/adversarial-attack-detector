import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── 1. Device setup ───────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── 2. Define CNN architecture ────────────────────────────────────────
class VictimCNN(nn.Module):
    def __init__(self):
        super(VictimCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── 3. Train function ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ── 4. Evaluate function ──────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# ── Only runs when you directly run: python3 train_classifier.py ──────
if __name__ == '__main__':
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    model     = VictimCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 50
    train_accs, test_accs = [], []

    print("Training VictimCNN on MNIST...")
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc        = evaluate(model, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss:.4f} | "
              f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    torch.save(model.state_dict(), 'models/victim_cnn.pth')
    print("\n✅ Model saved to models/victim_cnn.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), [a*100 for a in train_accs], label='Train Accuracy')
    plt.plot(range(1, EPOCHS+1), [a*100 for a in test_accs],  label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('VictimCNN Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/training_curve.png', dpi=120)
    print("✅ Graph saved to results/training_curve.png")
    print(f"\nFinal test accuracy: {test_accs[-1]*100:.2f}%")