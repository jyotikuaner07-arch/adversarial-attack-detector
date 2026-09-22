import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import io
import random
from PIL import Image, ImageFilter

# ── 1. Device setup ───────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ── 2. Real-world augmentation ────────────────────────────────────────
# NEW — added to make model robust to:
#   JPEG compression, blur, contrast variation
# Applied only at training time. Fires with 25% probability each.
class RealWorldDigitAugmentation:
    def __call__(self, image):
        # Mild blur (camera shake, low-res scan)
        if random.random() < 0.25:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8))
            )
        # Mild contrast variation (different screens, lighting)
        if random.random() < 0.25:
            image = transforms.functional.adjust_contrast(
                image, contrast_factor=random.uniform(0.75, 1.25)
            )
        # JPEG compression (Google images, screenshots, downloads)
        if random.random() < 0.25:
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=random.randint(60, 95))
            buf.seek(0)
            image = Image.open(buf).convert('L')
        return image


# ── 3. Define CNN architecture ────────────────────────────────────────
# UNCHANGED from original
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


# ── 4. Train function ─────────────────────────────────────────────────
# UNCHANGED from original
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


# ── 5. Evaluate function ──────────────────────────────────────────────
# UNCHANGED from original
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

    # NEW — training transform now includes:
    #   RandomAffine: handles off-centre, rotated, scaled digits
    #   RealWorldDigitAugmentation: handles JPEG/blur/contrast
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=12,             # rotate up to ±12°
            translate=(0.12, 0.12), # shift up to 12% in x and y
            scale=(0.85, 1.15),     # zoom in/out
            shear=8                 # slight perspective lean
        ),
        RealWorldDigitAugmentation(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Test transform — NO augmentation — same as original
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True,  download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform)

    # batch_size bumped to 256 for M1 speed
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0)

    model     = VictimCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # NEW — early stopping added (patience=5)
    # Augmentation makes training harder so we let it run up to 50
    # but stop early if validation stops improving
    train_accs, test_accs = [], []
    best_test_acc = 0
    patience      = 5
    no_improve    = 0
    EPOCHS        = 25

    print("Training VictimCNN on MNIST (with augmentation)...")
    print("Note: expect ~98-99% accuracy instead of 99.3% — normal with augmentation")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc        = evaluate(model, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss:.4f} | "
              f"Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'models/victim_cnn.pth')
            no_improve = 0
            print(f"           ✅ New best: {test_acc*100:.2f}% — saved")
        else:
            no_improve += 1
            print(f"           No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"   Best test accuracy: {best_test_acc*100:.2f}%")
            break

    actual_epochs = len(train_accs)

    print(f"\n✅ Model saved to models/victim_cnn.pth")

    # Graph — updated to use actual_epochs instead of fixed EPOCHS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, actual_epochs+1), [a*100 for a in train_accs], label='Train Accuracy')
    plt.plot(range(1, actual_epochs+1), [a*100 for a in test_accs],  label='Test Accuracy')
    plt.axhline(y=best_test_acc*100, color='green', linestyle=':',
                label=f'Best: {best_test_acc*100:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('VictimCNN Training Curve (with Augmentation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_curve.png', dpi=120)
    print("✅ Graph saved to results/training_curve.png")
    print(f"\nFinal test accuracy:  {test_accs[-1]*100:.2f}%")
    print(f"Best test accuracy:   {best_test_acc*100:.2f}%")
    print(f"Stopped at epoch:     {actual_epochs}")
    print()
    print("NEXT STEPS — run in this order:")
    print("  python3 generate_attacks.py   ← regenerate attacks on new model")
    print("  python3 train_detector.py     ← retrain detector on new attacks")
    print("  python3 app.py                ← start Flask")