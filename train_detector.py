import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

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
    from sklearn.metrics import classification_report
    print(f"Using device: {device}")
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ── 2. Load clean test images ─────────────────────────────────────
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=False, transform=transform)

    # ── 3. Load successful adversarial images from Phase 2 ───────────
    adv_path = 'attacks/detector_adv_success.pt'
    if not os.path.exists(adv_path):
        print("❌ ERROR: attacks/detector_adv_success.pt is missing.")
        sys.exit("run generate_attacks.py first")

    adv_images = torch.load(adv_path)
    num_adv = len(adv_images)
    print(f"✅ Loaded {num_adv} successful adversarial images from {adv_path}")

    if num_adv == 0:
        print("❌ ERROR: attacks/detector_adv_success.pt contains 0 samples.")
        sys.exit("run generate_attacks.py first")

    # Optional for transfer evaluation: load PGD and DeepFool bulk tensors if present
    pgd_adv = torch.load('attacks/PGD_adv_images.pt') if os.path.exists('attacks/PGD_adv_images.pt') else None
    df_adv  = torch.load('attacks/DeepFool_adv_images.pt') if os.path.exists('attacks/DeepFool_adv_images.pt') else None

    # ── 4. Build balanced training dataset ────────────────────────────
    # Balanced: exactly N clean MNIST vs N successful adv (unnormalized [0, 1])
    clean_subset = torch.stack([test_data[i % len(test_data)][0] for i in range(num_adv)])
    X_all = torch.cat([clean_subset, adv_images])
    y_all = torch.cat([
        torch.zeros(num_adv),   # clean  → label 0
        torch.ones(num_adv)     # adversarial → label 1
    ]).long()

    # Shuffle and split 80% train / 20% validation
    total_samples = len(X_all)
    idx = torch.randperm(total_samples)
    split = int(0.8 * total_samples)

    X_train = X_all[idx[:split]]
    y_train = y_all[idx[:split]]
    X_val   = X_all[idx[split:]]
    y_val   = y_all[idx[split:]]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

    print(f"✅ Balanced Dataset: {len(X_train)} train, {len(X_val)} validation ({num_adv} clean + {num_adv} adv)")

    # ── 5. Setup model, loss, optimiser ──────────────────────────────
    detector  = DetectorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=0.001)

    # ── 6. Training loop with early stopping ─────────────────────────
    EPOCHS = 30
    PATIENCE = 8
    patience_count = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    train_accs, val_accs = [], []

    print(f"\nTraining DetectorCNN for up to {EPOCHS} epochs (patience={PATIENCE})...")
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
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out  = detector(X)
                loss = criterion(out, y)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total   += len(y)
                val_loss    += loss.item() * len(y)
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / val_total

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(detector.state_dict(), 'models/detector_cnn.pth')
            saved = "✅ saved"
            patience_count = 0
        else:
            patience_count += 1
            saved = f"(no improve {patience_count}/{PATIENCE})"

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% (loss: {val_loss:.4f}) {saved}")

        if patience_count >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch} (best val acc: {best_val_acc:.1f}%)")
            break

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
    transfer_candidates = []
    if pgd_adv is not None:
        transfer_candidates.append(('PGD', pgd_adv))
    if df_adv is not None:
        transfer_candidates.append(('DeepFool', df_adv))

    for attack_name, adv_imgs in transfer_candidates:
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

    # Recall on DeepFool successful subset if available
    df_succ_recall = None
    if os.path.exists('attacks/DeepFool_index.json') and os.path.exists('attacks/DeepFool_adv_images.pt'):
        try:
            with open('attacks/DeepFool_index.json', 'r') as f_df:
                df_idx_data = json.load(f_df)
            df_succ_indices = [item['index'] for item in df_idx_data if item.get('success')]
            if df_succ_indices:
                df_bulk = torch.load('attacks/DeepFool_adv_images.pt')
                df_succ_imgs = torch.stack([df_bulk[i] for i in df_succ_indices])
                df_succ_labels = torch.ones(len(df_succ_imgs)).long()
                df_succ_loader = DataLoader(TensorDataset(df_succ_imgs, df_succ_labels), batch_size=64)
                df_succ_correct = 0
                with torch.no_grad():
                    for X, y in df_succ_loader:
                        X = X.to(device)
                        preds = detector(X).argmax(1).cpu()
                        df_succ_correct += (preds == y).sum().item()
                df_succ_recall = df_succ_correct / len(df_succ_imgs) * 100
                print(f"\n  🎯 DeepFool Successful Subset Recall: {df_succ_recall:.1f}% "
                      f"({df_succ_correct}/{len(df_succ_imgs)} correctly flagged as adversarial)")
        except Exception as e:
            print(f"  (DeepFool successful subset evaluation skipped: {e})")

    # ── 9. Also test on 200 clean images ──────────────────────────────
    # False positive rate: how often does detector wrongly flag clean images?
    clean_eval = torch.stack([test_data[i][0] for i in range(200)])
    clean_labels = torch.zeros(len(clean_eval)).long()
    clean_loader = DataLoader(
        TensorDataset(clean_eval, clean_labels),
        batch_size=64
    )
    clean_correct = 0
    with torch.no_grad():
        for X, y in clean_loader:
            X = X.to(device)
            preds = detector(X).argmax(1).cpu()
            clean_correct += (preds == y).sum().item()
    clean_acc = clean_correct / len(clean_eval) * 100
    false_pos = 100.0 - clean_acc

    print(f"\n  Clean image accuracy (200 test images): {clean_acc:.1f}%")
    print(f"  False positive rate:  {false_pos:.1f}% "
          f"(clean images wrongly flagged as adversarial)")
    if false_pos > 15.0:
        print(f"  ⚠️ HARD WARNING: False Positive Rate ({false_pos:.1f}%) exceeds 15%!")
    else:
        print(f"  ✅ False Positive Rate is within acceptable threshold ({false_pos:.1f}% <= 15%)")

    # ── 10. Final summary ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("📊 FINAL DETECTOR SUMMARY")
    print("="*55)
    print(f"  Trained on:              {num_adv} clean + {num_adv} successful adv images")
    print(f"  Validation accuracy:     {best_val_acc:.1f}%")
    if 'PGD' in transfer_results:
        print(f"  PGD transfer detection:  {transfer_results['PGD']:.1f}%")
    if 'DeepFool' in transfer_results:
        print(f"  DeepFool transfer:       {transfer_results['DeepFool']:.1f}%")
    if df_succ_recall is not None:
        print(f"  DeepFool success recall: {df_succ_recall:.1f}%")
    print(f"  Clean FPR (200 images):  {false_pos:.1f}%")
    print("="*55)

    # ── 11. Save training curve ───────────────────────────────────────
    plt.figure(figsize=(8, 5))
    epochs_run = len(train_accs)
    plt.plot(range(1, epochs_run + 1), train_accs,
             label='Train Accuracy', linewidth=2)
    plt.plot(range(1, epochs_run + 1), val_accs,
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