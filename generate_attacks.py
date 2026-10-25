import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import foolbox
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import argparse

ATTACK_INDICES = range(0, 500)

parser = argparse.ArgumentParser(description="Generate adversarial attacks on MNIST test images.")
parser.add_argument('--eval-split', action='store_true', help='Generate attacks on held-out test split (indices 500-1000)')
args = parser.parse_args()

if args.eval_split:
    ATTACK_INDICES = range(500, 1000)
    suffix = "_heldout"
else:
    suffix = ""

# ── 1. Device setup ───────────────────────────────────────────────────
device = torch.device('cpu')  # Foolbox does not support MPS — use CPU
print(f"Using device: {device}")
os.makedirs('attacks', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── 2. Load the trained victim model ─────────────────────────────────
from train_classifier import VictimCNN

model = VictimCNN().to(device)
model.load_state_dict(torch.load('models/victim_cnn.pth',
                                  map_location=device))
model.eval()
print("✅ Victim model loaded from models/victim_cnn.pth")

# ── 3. Load test images ───────────────────────────────────────────────
# ToTensor() only — NO Normalize here.
# Foolbox handles normalisation via preprocessing dict.
transform = transforms.ToTensor()
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=False, transform=transform)

attack_indices_list = list(ATTACK_INDICES)
images = torch.stack([test_data[i][0] for i in ATTACK_INDICES]).to(device)
labels = torch.tensor([test_data[i][1] for i in ATTACK_INDICES]).to(device)
print(f"✅ Loaded {len(images)} test images to attack (indices {attack_indices_list[0]} to {attack_indices_list[-1]})")

# ── 4. Baseline accuracy check ────────────────────────────────────────
# Normalise the same way training used
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))
norm_images = torch.stack([NORMALISE(images[i]) for i in range(len(images))])
with torch.no_grad():
    outputs = model(norm_images)
    baseline_acc = (outputs.argmax(1) == labels).float().mean().item()
print(f"✅ Baseline accuracy (no attack): {baseline_acc*100:.1f}%")

if baseline_acc < 0.95:
    print("⚠️  WARNING: Baseline accuracy is low (<95%).")
    print("   This means the model may not be loaded correctly.")
    print("   Check that models/victim_cnn.pth is from the latest training run.")

# ── 5. Wrap model in Foolbox ──────────────────────────────────────────
# preprocessing tells Foolbox to normalise images the same way
# the model was trained — this is critical for attacks to work correctly
preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
fmodel = foolbox.models.PyTorchModel(
    model,
    bounds=(0, 1),
    preprocessing=preprocessing
)
print("✅ Model wrapped in Foolbox")

# ── 6. Epsilon values ─────────────────────────────────────────────────
epsilons = [0.03, 0.05, 0.1, 0.15, 0.3]

# ── 7. Attacks ────────────────────────────────────────────────────────
attacks = {
    'FGSM':     fa.FGSM(),
    'PGD':      fa.LinfPGD(),
    'DeepFool': fa.LinfDeepFoolAttack(),
}

# ── 8. Run each attack ────────────────────────────────────────────────
results = {}
successful_samples = {}

for attack_name, attack in attacks.items():
    print(f"\n🔥 Running {attack_name} attack on {len(images)} images...")

    raw_advs, clipped_advs, success = attack(
        fmodel, images, labels, epsilons=epsilons
    )

    results[attack_name] = {
        'adv_images': clipped_advs,
        'success':    success,
    }

    print(f"  {'Epsilon':<10} {'Success Rate':<15} {'Meaning'}")
    print(f"  {'-'*45}")
    for i, eps in enumerate(epsilons):
        rate = success[i].float().mean().item() * 100
        meaning = "weak attack" if eps < 0.1 else "strong attack"
        print(f"  ε={eps:<8} {rate:.1f}%{'':<10} {meaning}")

    # 1. Save bulk adversarial images at epsilon=0.3
    save_tensor = clipped_advs[-1].cpu()
    pt_path = f'attacks/{attack_name}_adv_images{suffix}.pt'
    torch.save(save_tensor, pt_path)
    print(f"  ✅ Saved bulk tensor: {pt_path}")

    # 2. Predict on adversarial images to find successful misclassifications
    adv_norm = torch.stack([NORMALISE(save_tensor[i]) for i in range(len(save_tensor))]).to(device)
    with torch.no_grad():
        adv_preds = model(adv_norm).argmax(1).cpu()

    # 3. Build attacks/{ATTACK}_index.json and save tagged PNGs for successful attacks
    os.makedirs(f'attacks/{attack_name}', exist_ok=True)
    succ_list = []
    labels_cpu = labels.cpu()
    index_records = []

    for i in range(len(images)):
        real_idx = int(attack_indices_list[i])
        y = int(labels_cpu[i].item())
        p = int(adv_preds[i].item())
        is_succ = bool(success[-1][i].item() and p != y)

        index_records.append({
            "index": real_idx,
            "true": y,
            "adv_pred": p,
            "success": is_succ,
            "attack": attack_name,
            "eps": 0.3
        })

        if is_succ:
            succ_list.append(i)
            img_np = (save_tensor[i].squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            png_name = f"{attack_name}_true{y}_pred{p}_idx{real_idx:04d}{suffix}.png"
            json_name = f"{attack_name}_true{y}_pred{p}_idx{real_idx:04d}{suffix}.json"
            png_path = os.path.join('attacks', attack_name, png_name)
            json_path = os.path.join('attacks', attack_name, json_name)

            Image.fromarray(img_np, mode='L').save(png_path)
            with open(json_path, 'w') as jf:
                json.dump({
                    "attack": attack_name,
                    "eps": 0.3,
                    "true": y,
                    "adv_pred": p,
                    "index": real_idx
                }, jf, indent=2)

    # Save index JSON containing full list of attempted samples
    json_path = f'attacks/{attack_name}_index{suffix}.json'
    with open(json_path, 'w') as f_idx:
        json.dump(index_records, f_idx, indent=2)
    print(f"  ✅ Saved {attack_name} index: {json_path} ({len(index_records)} records)")

    successful_samples[attack_name] = succ_list
    print(f"  ✅ Saved {len(succ_list)} tagged PNGs & JSON sidecars to attacks/{attack_name}/")

# ── Save detector training tensor (successful FGSM, PGD, and DeepFool) ──
if not args.eval_split:
    detector_tensors = []
    detector_meta = []

    for atk in ['FGSM', 'PGD', 'DeepFool']:
        if atk in successful_samples and atk in results:
            adv_tensor = results[atk]['adv_images'][-1].cpu()
            adv_norm = torch.stack([NORMALISE(adv_tensor[i]) for i in range(len(adv_tensor))]).to(device)
            with torch.no_grad():
                preds = model(adv_norm).argmax(1).cpu()
            labels_cpu = labels.cpu()
            for i in successful_samples[atk]:
                detector_tensors.append(adv_tensor[i])
                detector_meta.append({
                    "attack": atk,
                    "eps": 0.3,
                    "true": int(labels_cpu[i].item()),
                    "adv_pred": int(preds[i].item()),
                    "index": int(attack_indices_list[i])
                })

    if detector_tensors:
        detector_dataset = torch.stack(detector_tensors)
        torch.save(detector_dataset, 'attacks/detector_adv_success.pt')
        with open('attacks/detector_adv_meta.json', 'w') as f:
            json.dump(detector_meta, f, indent=2)
        print(f"\n✅ Detector training dataset saved to attacks/detector_adv_success.pt ({len(detector_tensors)} samples)")
        print(f"✅ Detector metadata saved to attacks/detector_adv_meta.json")
    else:
        print("\n⚠️ WARNING: No successful attacks found to save for detector!")

# Print attempted vs successful summary
print("\n" + "="*55)
print("📊 ATTACK SUCCESS COUNTS (at ε=0.3)")
print("="*55)
for name in attacks.keys():
    succ_cnt = len(successful_samples.get(name, []))
    total_cnt = len(images)
    print(f"  {name:<10}: {succ_cnt}/{total_cnt} successful attacks ({succ_cnt/total_cnt*100:.1f}%)")
    if name == 'FGSM' and (succ_cnt / total_cnt) < 0.5:
        print("  👉 NOTE: FGSM success rate is lower on the augmented victim; use tagged PGD or DeepFool PNGs in attacks/PGD/ or attacks/DeepFool/ for demo uploads (e.g. 7→2→7 style recovery).")
print("="*55)

# ── 9. VALIDATION — confirm saved attacks actually fool the model ─────
print("\n" + "="*55)
print("🔍 VALIDATING SAVED ATTACK FILES")
print("="*55)

all_valid = True

for attack_name in attacks.keys():
    path = f'attacks/{attack_name}_adv_images{suffix}.pt'
    adv_tensors = torch.load(path)

    # Normalise the adversarial images before passing to model
    adv_norm = torch.stack([NORMALISE(adv_tensors[i]) for i in range(len(adv_tensors))])

    with torch.no_grad():
        outputs    = model(adv_norm)
        preds      = outputs.argmax(1)
        true_labels_cpu = labels.cpu()
        preds_cpu       = preds.cpu()

    # Success = model predicted wrong class
    fooled      = (preds_cpu != true_labels_cpu).float()
    success_rate = fooled.mean().item() * 100

    # Check if model is predicting only one class (broken attack symptom)
    unique_preds = preds_cpu.unique()
    pred_spread  = len(unique_preds)

    print(f"\n  {attack_name}:")
    print(f"    Attack success rate: {success_rate:.1f}%")
    print(f"    Unique predicted classes: {sorted(unique_preds.tolist())}")
    print(f"    Number of distinct predictions: {pred_spread}")

    if pred_spread <= 2:
        print(f"    ❌ BROKEN — model only predicts {pred_spread} class(es).")
        print(f"       This means attacks are not working properly.")
        print(f"       Most likely cause: normalisation mismatch.")
        all_valid = False
    elif success_rate < 50:
        print(f"    ⚠️  LOW success rate — attacks may be weak.")
        all_valid = False
    else:
        print(f"    ✅ Valid — attacks are working correctly.")

print()
if all_valid:
    print("✅ All attack files validated — attacks are working correctly.")
    print("   You can now run: python3 train_detector.py")
else:
    print("❌ VALIDATION FAILED — attack files are not reliable.")
    print()
    print("   Most likely cause:")
    print("   The model was retrained but normalisation inside")
    print("   Foolbox does not match the model's training normalisation.")
    print()
    print("   Fix: Make sure preprocessing in Foolbox matches exactly:")
    print("   preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)")
    print()
    print("   Also verify your model's forward() does NOT apply")
    print("   normalisation internally — normalisation should only")
    print("   happen via Foolbox's preprocessing dict.")

# ── 10. Verify saved files ────────────────────────────────────────────
print("\n📁 Saved attack files:")
for f in sorted(os.listdir('attacks')):
    if f.endswith('.pt'):
        size = os.path.getsize(f'attacks/{f}') / (1024*1024)
        print(f"   {f} ({size:.1f} MB)")

# ── 11. Visualisation ─────────────────────────────────────────────────
print("\n🎨 Creating visualisation...")

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
attack_names = list(attacks.keys())

for col in range(5):
    orig = images[col].cpu().squeeze()
    axes[0, col].imshow(orig, cmap='gray')
    axes[0, col].set_title(f'Original\nLabel: {labels[col].item()}', fontsize=9)
    axes[0, col].axis('off')

    for row, name in enumerate(attack_names):
        adv = results[name]['adv_images'][-1][col].cpu().squeeze()
        axes[row+1, col].imshow(adv, cmap='gray')
        success_flag = results[name]['success'][-1][col].item()
        status = "✗ FOOLED" if success_flag else "✓ resisted"
        axes[row+1, col].set_title(f'{name}\n{status}', fontsize=9,
            color='red' if success_flag else 'green')
        axes[row+1, col].axis('off')

row_labels = ['Original', 'FGSM\n(ε=0.3)', 'PGD\n(ε=0.3)', 'DeepFool\n(ε=0.3)']
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=10, fontweight='bold',
                             rotation=0, labelpad=60, va='center')

plt.suptitle('Original vs Adversarial Examples\n'
             '(Images look identical to humans — AI is fooled)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
vis_path = f'results/attack_visualisation{suffix}.png'
plt.savefig(vis_path, dpi=120, bbox_inches='tight')
print(f"✅ Saved: {vis_path}")

# ── 12. Final summary ─────────────────────────────────────────────────
print("\n" + "="*55)
print("📊 ATTACK SUMMARY (at ε=0.3, strongest attack)")
print("="*55)
print(f"  Baseline accuracy (no attack):  {baseline_acc*100:.1f}%")
print(f"  {'Attack':<12} {'Success Rate':<15} {'Accuracy Left'}")
print(f"  {'-'*45}")
for name in attacks.keys():
    success_rate  = results[name]['success'][-1].float().mean().item()*100
    accuracy_left = 100 - success_rate
    print(f"  {name:<12} {success_rate:.1f}%{'':<10} {accuracy_left:.1f}%")
print("="*55)
print("\n✅ Phase 2 complete! All attack files saved to attacks/")
print("   You can now run Phase 3: python3 train_detector.py")