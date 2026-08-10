import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Import VictimCNN device setting from Phase 1
from train_classifier import device

# ── 1. Define DetectorCNN ─────────────────────────────────────────────
# This is the security guard network.
# Same architecture as VictimCNN but output = 2 classes
# (clean=0, adversarial=1) instead of 10 digit classes.
class DetectorCNN(nn.Module):
    def __init__(self):
        super(DetectorCNN, self).__init__()
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
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)   # 2 outputs: clean or adversarial
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── Only runs when directly executed ──────────────────────────────────
if __name__ == '__main__':
    print(f"Using device: {device}")
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ── 2. Load clean images ──────────────────────────────────────────
    # We use the same 500 test images that were attacked in Phase 2.
    # These are CLEAN (label = 0).
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=False, transform=transform)

    clean_images = torch.stack(
        [test_data[i][0] for i in range(500)])  # shape: [500, 1, 28, 28]

    # ── 3. Load adversarial images from Phase 2 ───────────────────────
    # FGSM images used for TRAINING the detector (label = 1)
    # PGD and DeepFool used ONLY for transfer detection testing
    fgsm_adv = torch.load('attacks/FGSM_adv_images.pt')    # [500, 1, 28, 28]
    pgd_adv  = torch.load('attacks/PGD_adv_images.pt')     # [500, 1, 28, 28]
    df_adv   = torch.load('attacks/DeepFool_adv_images.pt')# [500, 1, 28, 28]
    print("✅ Loaded clean and adversarial images")

    # ── 4. Build training dataset ─────────────────────────────────────
    # 500 clean (label=0) + 500 FGSM adversarial (label=1) = 1000 total
    # The detector is ONLY trained on FGSM — not PGD or DeepFool.
    # That's intentional — transfer detection tests if it can catch
    # attacks it was never trained on.
    X_all = torch.cat([clean_images, fgsm_adv])       # [1000, 1, 28, 28]
    y_all = torch.cat([
        torch.zeros(500),   # 500 clean  → label 0
        torch.ones(500)     # 500 FGSM   → label 1
    ]).long()

    # Shuffle and split 80% train / 20% validation
    idx     = torch.randperm(1000)
    split   = int(0.8 * 1000)  # 800 train, 200 validation

    X_train = X_all[idx[:split]]
    y_train = y_all[idx[:split]]
    X_val   = X_all[idx[split:]]
    y_val   = y_all[idx[split:]]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

    print(f"✅ Dataset: {len(X_train)} train, {len(X_val)} validation")
    print(f"   Training on: 500 clean + 500 FGSM adversarial images")

    # ── 5. Setup model, loss, optimiser ──────────────────────────────
    detector  = DetectorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=0.001)

    # ── 6. Training loop ──────────────────────────────────────────────
    EPOCHS = 100
    train_accs, val_accs = [], []
    best_val_acc = 0

    print(f"\nTraining DetectorCNN for {EPOCHS} epochs...")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):

        # Training phase
        detector.train()
        train_correct, train_total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = detector(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total   += len(y)
        train_acc = train_correct / train_total * 100

        # Validation phase
        detector.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out  = detector(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total   += len(y)
        val_acc = val_correct / val_total * 100

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(detector.state_dict(), 'models/detector_cnn.pth')
            saved = "✅ saved"
        else:
            saved = ""

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% {saved}")

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print("✅ Detector saved to models/detector_cnn.pth")

    # ── 7. Detailed evaluation on validation set ──────────────────────
    print("\n" + "="*55)
    print("📊 DETAILED VALIDATION REPORT")
    print("="*55)

    detector.load_state_dict(torch.load('models/detector_cnn.pth',
                                         map_location=device))
    detector.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            preds = detector(X).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print(classification_report(
        all_labels, all_preds,
        target_names=['Clean', 'Adversarial'],
        digits=3
    ))

    # ── 8. Transfer Detection ─────────────────────────────────────────
    # This is the key research result.
    # Detector was trained ONLY on FGSM.
    # Now we test it against PGD and DeepFool — attacks it never saw.
    print("="*55)
    print("🧪 TRANSFER DETECTION TEST")
    print("   Detector trained on FGSM only.")
    print("   Testing against attacks it has NEVER seen:")
    print("="*55)

    transfer_results = {}

    for attack_name, adv_imgs in [('PGD', pgd_adv), ('DeepFool', df_adv)]:
        # All adversarial → label should be 1
        adv_labels  = torch.ones(len(adv_imgs)).long()
        adv_loader  = DataLoader(
            TensorDataset(adv_imgs, adv_labels),
            batch_size=64
        )

        correct = 0
        all_p, all_l = [], []
        with torch.no_grad():
            for X, y in adv_loader:
                X = X.to(device)
                preds = detector(X).argmax(1).cpu()
                correct += (preds == y).sum().item()
                all_p.extend(preds.tolist())
                all_l.extend(y.tolist())

        acc = correct / len(adv_imgs) * 100
        transfer_results[attack_name] = acc
        print(f"\n  {attack_name} Transfer Detection: {acc:.1f}%")
        print(f"  (detector correctly flagged {correct}/{len(adv_imgs)} "
              f"{attack_name} images as adversarial)")

    # ── 9. Also test on clean images ──────────────────────────────────
    # False positive rate: how often does detector wrongly flag clean images?
    clean_labels = torch.zeros(500).long()
    clean_loader = DataLoader(
        TensorDataset(clean_images, clean_labels),
        batch_size=64
    )
    clean_correct = 0
    with torch.no_grad():
        for X, y in clean_loader:
            X = X.to(device)
            preds = detector(X).argmax(1).cpu()
            clean_correct += (preds == y).sum().item()
    clean_acc = clean_correct / 500 * 100
    false_pos  = 100 - clean_acc

    print(f"\n  Clean image accuracy: {clean_acc:.1f}%")
    print(f"  False positive rate:  {false_pos:.1f}% "
          f"(clean images wrongly flagged as adversarial)")

    # ── 10. Final summary ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("📊 FINAL DETECTOR SUMMARY")
    print("="*55)
    print(f"  Trained on:              FGSM (500 images)")
    print(f"  Validation accuracy:     {best_val_acc:.1f}%")
    print(f"  PGD transfer detection:  {transfer_results['PGD']:.1f}%")
    print(f"  DeepFool transfer:       {transfer_results['DeepFool']:.1f}%")
    print(f"  False positive rate:     {false_pos:.1f}%")
    print("="*55)

    # ── 11. Save training curve ───────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), train_accs,
             label='Train Accuracy', linewidth=2)
    plt.plot(range(1, EPOCHS+1), val_accs,
             label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.axhline(y=best_val_acc, color='green', linestyle=':',
                label=f'Best Val: {best_val_acc:.1f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('DetectorCNN Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/detector_training_curve.png', dpi=120)
    print("\n✅ Curve saved to results/detector_training_curve.png")
    print("\n✅ Phase 3 complete! Run Phase 4: python3 defence.py")
    print("   Then Phase 5: python3 app.py")