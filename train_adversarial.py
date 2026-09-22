import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from train_classifier import VictimCNN

# ── 1. Device setup ───────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Normalization applied INSIDE forward path only
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))


def forward_norm(model, x):
    """Pass unnormalized [0, 1] images through Normalize then VictimCNN."""
    return model(NORMALISE(x))


# ── 2. Fast FGSM (Wong et al.) ─────────────────────────────────────────
def fast_fgsm_batch(model, x, y, eps=0.3, alpha=0.375):
    """
    Fast Adversarial Training (Wong et al., 2020):
    Random initialization within [-eps, eps], followed by single-step FGSM.
    Operates strictly in [0, 1] pixel space.
    """
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = torch.clamp(x + delta, 0.0, 1.0) - x
    delta.requires_grad = True

    out = forward_norm(model, x + delta)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grad = delta.grad.detach()
    delta = delta.detach() + alpha * grad.sign()
    delta = torch.clamp(delta, -eps, eps)
    delta = torch.clamp(x + delta, 0.0, 1.0) - x
    return (x + delta).detach()


# ── 3. Evaluation helpers ─────────────────────────────────────────────
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = forward_norm(model, images)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def evaluate_fgsm(model, loader, eps=0.3):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images_adv = images.clone().detach()
        images_adv.requires_grad = True

        out = forward_norm(model, images_adv)
        loss = F.cross_entropy(out, labels)
        loss.backward()

        with torch.no_grad():
            adv = images_adv + eps * images_adv.grad.sign()
            adv = torch.clamp(adv, 0.0, 1.0)
            out_adv = forward_norm(model, adv)
            correct += (out_adv.argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def evaluate_pgd(model, loader, eps=0.3, alpha=0.01, steps=20):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        delta = torch.zeros_like(images).uniform_(-eps, eps)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images

        for _ in range(steps):
            delta.requires_grad = True
            out = forward_norm(model, images + delta)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            grad = delta.grad.detach()
            delta = delta.detach() + alpha * grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(images + delta, 0.0, 1.0) - images

        with torch.no_grad():
            adv = images + delta
            out_adv = forward_norm(model, adv)
            correct += (out_adv.argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


# ── 4. Main training routine ──────────────────────────────────────────
def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load unnormalized [0, 1] MNIST
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    # 1000 test samples for evaluation/early stopping validation
    val_subset = Subset(test_dataset, range(1000))
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=0)

    model = VictimCNN().to(device)

    # Warm start from existing models/victim_cnn.pth if present
    victim_path = 'models/victim_cnn.pth'
    if os.path.exists(victim_path):
        model.load_state_dict(torch.load(victim_path, map_location=device))
        print(f"✅ Warm-started weights from {victim_path}")
    else:
        print(f"⚠️ {victim_path} not found; initializing model from scratch.")

    # ── Evaluate BEFORE Adversarial Training ─────────────────────────
    print("\n" + "=" * 60)
    print("📊 EVALUATION BEFORE ADVERSARIAL TRAINING")
    print("=" * 60)
    print("Evaluating baseline on test subset (1000 samples)...")
    before_clean = evaluate_clean(model, val_loader)
    print(f"  Clean Accuracy:    {before_clean * 100:.2f}%")
    before_fgsm = evaluate_fgsm(model, val_loader, eps=0.3)
    print(f"  FGSM (ε=0.3):      {before_fgsm * 100:.2f}%")
    before_pgd = evaluate_pgd(model, val_loader, eps=0.3, alpha=0.01, steps=20)
    print(f"  PGD-20 (ε=0.3):    {before_pgd * 100:.2f}%")
    print("=" * 60 + "\n")

    # Optimizer and hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    EPS = 0.3
    ALPHA = 1.25 * EPS  # 0.375
    MAX_EPOCHS = 15
    PATIENCE = 4

    best_pgd_acc = before_pgd
    patience_count = 0
    save_path = 'models/victim_cnn_at.pth'

    print(f"Starting Fast Adversarial Training (Wong et al.):")
    print(f"  Max Epochs: {MAX_EPOCHS} | Patience: {PATIENCE} (on PGD-20)")
    print(f"  Epsilon: {EPS} | Alpha: {ALPHA}")
    print(f"  Loss: 0.5 * CE(clean) + 0.5 * CE(adv)")
    print(f"  Target Save Path: {save_path} (DO NOT overwrite victim_cnn.pth)")
    print("-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, correct_clean, correct_adv, total_samples = 0.0, 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Generate Fast FGSM perturbation
            images_adv = fast_fgsm_batch(model, images, labels, eps=EPS, alpha=ALPHA)

            optimizer.zero_grad()
            out_clean = forward_norm(model, images)
            out_adv = forward_norm(model, images_adv)

            loss_clean = criterion(out_clean, labels)
            loss_adv = criterion(out_adv, labels)
            loss = 0.5 * loss_clean + 0.5 * loss_adv

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct_clean += (out_clean.argmax(1) == labels).sum().item()
            correct_adv += (out_adv.argmax(1) == labels).sum().item()
            total_samples += len(labels)

        train_loss = total_loss / total_samples
        train_clean_acc = correct_clean / total_samples
        train_adv_acc = correct_adv / total_samples

        # Validation evaluation
        val_clean = evaluate_clean(model, val_loader)
        val_fgsm = evaluate_fgsm(model, val_loader, eps=EPS)
        val_pgd = evaluate_pgd(model, val_loader, eps=EPS, alpha=0.01, steps=20)

        saved = ""
        if val_pgd > best_pgd_acc:
            best_pgd_acc = val_pgd
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            saved = "✅ Saved new best AT model"
        else:
            patience_count += 1
            saved = f"({patience_count}/{PATIENCE})"

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | Loss: {train_loss:.4f} | "
              f"Clean: {val_clean*100:.1f}% | FGSM: {val_fgsm*100:.1f}% | "
              f"PGD-20: {val_pgd*100:.1f}% | {saved}")

        if patience_count >= PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch} (best PGD-20: {best_pgd_acc*100:.2f}%)")
            break

    print("\n" + "=" * 60)
    print("🏁 ADVERSARIAL TRAINING COMPLETE")
    print(f"  Best PGD-20 Accuracy: {best_pgd_acc*100:.2f}%")
    print(f"  Model saved ONLY to:  {save_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
