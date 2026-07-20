import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import foolbox
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ── 1. Device setup ───────────────────────────────────────────────────
device = torch.device('cpu')  # Foolbox does not support MPS — use CPU
print(f"Using device: {device}")
os.makedirs('attacks', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── 2. Load the trained victim model ─────────────────────────────────
# We import the VictimCNN class we already defined in Phase 1
from train_classifier import VictimCNN

model = VictimCNN().to(device)
model.load_state_dict(torch.load('models/victim_cnn.pth',
                                  map_location=device))
model.eval()
print("✅ Victim model loaded from models/victim_cnn.pth")

# ── 3. Load test images to attack ────────────────────────────────────
# We use ToTensor() only — NO Normalize here.
# Foolbox handles normalisation internally via preprocessing dict below.
transform = transforms.ToTensor()
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=False, transform=transform)

# Take first 500 images and labels
images = torch.stack([test_data[i][0] for i in range(500)]).to(device)
labels = torch.tensor([test_data[i][1] for i in range(500)]).to(device)
print(f"✅ Loaded {len(images)} test images to attack")

# ── 4. Check baseline accuracy (before any attack) ───────────────────
# This confirms our model works correctly before we attack it
from torchvision.transforms.functional import normalize
norm_images = normalize(images, mean=[0.1307], std=[0.3081])
with torch.no_grad():
    outputs = model(norm_images)
    baseline_acc = (outputs.argmax(1) == labels).float().mean().item()
print(f"✅ Baseline accuracy (no attack): {baseline_acc*100:.1f}%")

# ── 5. Wrap model in Foolbox ──────────────────────────────────────────
# Foolbox needs to know:
# - bounds: pixel values range from 0 to 1
# - preprocessing: the same Normalize we applied during training
preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
fmodel = foolbox.models.PyTorchModel(
    model,
    bounds=(0, 1),
    preprocessing=preprocessing
)
print("✅ Model wrapped in Foolbox")

# ── 6. Define epsilon values ──────────────────────────────────────────
# Epsilon controls attack strength — how much noise is added
# 0.03 = barely visible   0.3 = still looks same to humans but model fails
epsilons = [0.03, 0.05, 0.1, 0.15, 0.3]

# ── 7. Define the 3 attacks ───────────────────────────────────────────
attacks = {
    'FGSM':     fa.FGSM(),               # Fast, single step
    'PGD':      fa.LinfPGD(),            # Stronger, multi-step
    'DeepFool': fa.LinfDeepFoolAttack(), # Minimal perturbation
}

# ── 8. Run each attack and save results ───────────────────────────────
results = {}

for attack_name, attack in attacks.items():
    print(f"\n🔥 Running {attack_name} attack on {len(images)} images...")

    # Run the attack
    # Returns: (clipped_advs, advs, success)
    # - advs: the actual adversarial images for each epsilon
    # - success: True/False for each image at each epsilon
    raw_advs, clipped_advs, success = attack(
        fmodel, images, labels, epsilons=epsilons
    )

    # success shape: [num_epsilons, num_images]
    # We store results for all epsilons
    results[attack_name] = {
        'adv_images': clipped_advs,   # list of tensors, one per epsilon
        'success':    success,         # boolean tensor
    }

    # Print success rate at each epsilon
    print(f"  {'Epsilon':<10} {'Success Rate':<15} {'Meaning'}")
    print(f"  {'-'*45}")
    for i, eps in enumerate(epsilons):
        rate = success[i].float().mean().item() * 100
        meaning = "weak attack" if eps < 0.1 else "strong attack"
        print(f"  ε={eps:<8} {rate:.1f}%{'':<10} {meaning}")

    # Save adversarial images at epsilon=0.3 (strongest, used for training)
    save_tensor = clipped_advs[-1].cpu()  # -1 = last epsilon = 0.3
    torch.save(save_tensor, f'attacks/{attack_name}_adv_images.pt')
    print(f"  ✅ Saved: attacks/{attack_name}_adv_images.pt")

# ── 9. Verify saved files ─────────────────────────────────────────────
print("\n📁 Saved attack files:")
for f in os.listdir('attacks'):
    size = os.path.getsize(f'attacks/{f}') / (1024*1024)
    print(f"   {f} ({size:.1f} MB)")

# ── 10. Visualise: original vs all 3 attacks ──────────────────────────
print("\n🎨 Creating visualisation...")

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
attack_names = list(attacks.keys())

# Row 0 = Original images
# Row 1 = FGSM adversarial
# Row 2 = PGD adversarial
# Row 3 = DeepFool adversarial

for col in range(5):  # show 5 different digits
    # Original image
    orig = images[col].cpu().squeeze()
    axes[0, col].imshow(orig, cmap='gray')
    axes[0, col].set_title(f'Original\nLabel: {labels[col].item()}',
                            fontsize=9)
    axes[0, col].axis('off')

    # Each attack type
    for row, name in enumerate(attack_names):
        adv = results[name]['adv_images'][-1][col].cpu().squeeze()
        # Calculate perturbation (difference between original and attacked)
        perturbation = (adv - orig).abs()

        axes[row+1, col].imshow(adv, cmap='gray')
        success_flag = results[name]['success'][-1][col].item()
        status = "✗ FOOLED" if success_flag else "✓ resisted"
        axes[row+1, col].set_title(f'{name}\n{status}', fontsize=9,
            color='red' if success_flag else 'green')
        axes[row+1, col].axis('off')

# Row labels on left side
row_labels = ['Original', 'FGSM\n(ε=0.3)', 'PGD\n(ε=0.3)', 'DeepFool\n(ε=0.3)']
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=10, fontweight='bold',
                             rotation=0, labelpad=60, va='center')

plt.suptitle('Original vs Adversarial Examples\n'
             '(Images look identical to humans — AI is fooled)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/attack_visualisation.png', dpi=120,
            bbox_inches='tight')
print("✅ Saved: results/attack_visualisation.png")

# ── 11. Final summary table ───────────────────────────────────────────
print("\n" + "="*55)
print("📊 ATTACK SUMMARY (at ε=0.3, strongest attack)")
print("="*55)
print(f"  Baseline accuracy (no attack):  {baseline_acc*100:.1f}%")
print(f"  {'Attack':<12} {'Success Rate':<15} {'Accuracy Left'}")
print(f"  {'-'*45}")
for name in attacks.keys():
    success_rate = results[name]['success'][-1].float().mean().item()*100
    accuracy_left = 100 - success_rate
    print(f"  {name:<12} {success_rate:.1f}%{'':<10} {accuracy_left:.1f}%")
print("="*55)
print("\n✅ Phase 2 complete! All attack files saved to attacks/")
print("   You can now run Phase 3: python3 train_detector.py")