import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

# Import VictimCNN class from Phase 1
from train_classifier import VictimCNN

# ── 1. Device & model setup ───────────────────────────────────────────
device = torch.device('cpu')  # keep on CPU for consistency

model = VictimCNN().to(device)
model.load_state_dict(torch.load('models/victim_cnn.pth',
                                  map_location=device))
model.eval()
print("✅ VictimCNN loaded from models/victim_cnn.pth")

NORMALISE = transforms.Normalize((0.1307,), (0.3081,))

# ── 2. Define defence functions ───────────────────────────────────────

def jpeg_defence(tensor_img, quality=75):
    """
    JPEG Compression Defence.
    Saves image as JPEG at given quality then reloads it.
    This strips high-frequency adversarial noise while preserving
    the digit shape that the model needs to classify correctly.
    Quality=100 = no compression. Quality=50 = heavy compression.
    """
    # tensor_img shape: [1, 1, 28, 28] — squeeze to [28, 28]
    img_np = (tensor_img.squeeze().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')  # L = greyscale

    # Save as JPEG into a memory buffer (not a real file)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    # Reload from buffer — this is the "compressed" image
    compressed = Image.open(buffer)
    compressed_tensor = transforms.ToTensor()(np.array(compressed))
    return compressed_tensor.unsqueeze(0)  # back to [1, 1, 28, 28]


def bit_depth_reduction(tensor_img, bits=4):
    """
    Bit-Depth Reduction Defence.
    Rounds pixel values to fewer discrete levels.
    Normal: 256 levels (8-bit). After: 16 levels (4-bit).
    The attacker crafted noise at exact fractional values —
    rounding destroys that precision.
    """
    steps = 2 ** bits - 1   # 4-bit = 15 steps, 5-bit = 31 steps
    return torch.round(tensor_img * steps) / steps


def gaussian_smoothing(tensor_img, sigma=1.0):
    """
    Gaussian Blur Defence.
    Applies slight blur to smooth out sharp adversarial noise.
    Uses a 3x3 Gaussian kernel.
    """
    import cv2
    img_np = tensor_img.squeeze().numpy()
    blurred = cv2.GaussianBlur(img_np, (3, 3), sigma)
    return torch.tensor(blurred).unsqueeze(0).unsqueeze(0).float()


# ── 3. Prediction helper ──────────────────────────────────────────────

def get_accuracy(images, labels, defence_fn=None, name="No defence"):
    """
    Run images through optional defence then through VictimCNN.
    Returns accuracy percentage.
    """
    correct = 0
    for i in range(len(images)):
        img = images[i:i+1].cpu()

        # Apply defence if provided
        if defence_fn is not None:
            img = defence_fn(img)

        # Normalise and predict
        norm = NORMALISE(img.squeeze(0)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(norm).argmax().item()

        if pred == labels[i].item():
            correct += 1

    acc = correct / len(images) * 100
    print(f"  {name:<40} {acc:.1f}%")
    return acc


# ── 4. Load adversarial images ────────────────────────────────────────
print("\nLoading adversarial images...")
transform = transforms.ToTensor()
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=False, transform=transform)

# Use first 200 images for speed
labels = torch.tensor([test_data[i][1] for i in range(200)])
clean  = torch.stack([test_data[i][0] for i in range(200)])

# Load adversarial files from Phase 2
fgsm_adv = torch.load('attacks/FGSM_adv_images.pt')[:200]
pgd_adv  = torch.load('attacks/PGD_adv_images.pt')[:200]
df_adv   = torch.load('attacks/DeepFool_adv_images.pt')[:200]
print("✅ Loaded 200 images from each attack type")

os.makedirs('results', exist_ok=True)

# ── 5. Run defence evaluation ─────────────────────────────────────────
print("\n" + "="*60)
print("DEFENCE EVALUATION RESULTS")
print("="*60)

# Store results for plotting
defence_names = []
fgsm_accs     = []
pgd_accs      = []
df_accs       = []

# ── Baseline: no attack ───────────────────────────────────────────────
print("\n📊 Baseline (no attack):")
baseline = get_accuracy(clean, labels, name="Clean images (no attack)")

# ── No defence: raw adversarial images ───────────────────────────────
print("\n📊 No defence (raw adversarial):")
get_accuracy(fgsm_adv, labels, name="FGSM — no defence")
get_accuracy(pgd_adv,  labels, name="PGD  — no defence")
get_accuracy(df_adv,   labels, name="DeepFool — no defence")

# ── JPEG compression defences ─────────────────────────────────────────
print("\n📊 JPEG Compression Defence:")
for quality in [75, 60, 50]:
    name = f"JPEG q={quality}"
    defence_names.append(name)
    fa = get_accuracy(fgsm_adv, labels,
         lambda x: jpeg_defence(x, quality), f"FGSM + JPEG q={quality}")
    pa = get_accuracy(pgd_adv,  labels,
         lambda x: jpeg_defence(x, quality), f"PGD  + JPEG q={quality}")
    da = get_accuracy(df_adv,   labels,
         lambda x: jpeg_defence(x, quality), f"DeepFool + JPEG q={quality}")
    fgsm_accs.append(fa)
    pgd_accs.append(pa)
    df_accs.append(da)

# ── Bit-depth reduction defences ─────────────────────────────────────
print("\n📊 Bit-Depth Reduction Defence:")
for bits in [4, 5, 6]:
    name = f"Bit-depth {bits}-bit"
    defence_names.append(name)
    fa = get_accuracy(fgsm_adv, labels,
         lambda x: bit_depth_reduction(x, bits), f"FGSM + {bits}-bit reduction")
    pa = get_accuracy(pgd_adv,  labels,
         lambda x: bit_depth_reduction(x, bits), f"PGD  + {bits}-bit reduction")
    da = get_accuracy(df_adv,   labels,
         lambda x: bit_depth_reduction(x, bits), f"DeepFool + {bits}-bit reduction")
    fgsm_accs.append(fa)
    pgd_accs.append(pa)
    df_accs.append(da)

# ── Gaussian smoothing defence ────────────────────────────────────────
print("\n📊 Gaussian Smoothing Defence:")
for sigma in [0.5, 1.0]:
    name = f"Gaussian σ={sigma}"
    defence_names.append(name)
    fa = get_accuracy(fgsm_adv, labels,
         lambda x: gaussian_smoothing(x, sigma), f"FGSM + Gaussian σ={sigma}")
    pa = get_accuracy(pgd_adv,  labels,
         lambda x: gaussian_smoothing(x, sigma), f"PGD  + Gaussian σ={sigma}")
    da = get_accuracy(df_adv,   labels,
         lambda x: gaussian_smoothing(x, sigma), f"DeepFool + Gaussian σ={sigma}")
    fgsm_accs.append(fa)
    pgd_accs.append(pa)
    df_accs.append(da)

# ── 6. Print summary table ────────────────────────────────────────────
print("\n" + "="*60)
print("📊 DEFENCE SUMMARY TABLE")
print(f"  Baseline (clean, no attack): {baseline:.1f}%")
print("="*60)
print(f"  {'Defence':<22} {'FGSM':>8} {'PGD':>8} {'DeepFool':>10}")
print(f"  {'-'*50}")
print(f"  {'No defence (raw attack)':<22} {'22.4%':>8} {'0.0%':>8} {'0.0%':>10}")
for i, name in enumerate(defence_names):
    print(f"  {name:<22} {fgsm_accs[i]:>7.1f}% {pgd_accs[i]:>7.1f}% {df_accs[i]:>9.1f}%")
print("="*60)

# ── 7. Clean plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig.suptitle('Accuracy Recovery After Defence\n(Baseline: 99% — Higher bar = better defence)',
             fontsize=13, fontweight='bold', y=1.02)

attack_labels = ['FGSM', 'PGD', 'DeepFool']
attack_data   = [fgsm_accs, pgd_accs, df_accs]
colors        = ['#3B82F6', '#EF4444', '#10B981']

for ax, atk_name, atk_accs, color in zip(axes, attack_labels, attack_data, colors):
    bars = ax.bar(defence_names, atk_accs, color=color, alpha=0.85, width=0.5)

    # Baseline line
    ax.axhline(y=baseline, color='black', linestyle='--',
               linewidth=1.5, label=f'Baseline: {baseline:.0f}%')

    # No-defence reference line
    no_def = {'FGSM': 22.4, 'PGD': 0.0, 'DeepFool': 0.0}
    ax.axhline(y=no_def[atk_name], color='red', linestyle=':',
               linewidth=1.2, label=f'No defence: {no_def[atk_name]:.0f}%')

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'{atk_name} Attack', fontsize=12, fontweight='bold', color=color)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Model Accuracy (%)' if ax == axes[0] else '')
    ax.set_xticklabels(defence_names, rotation=35, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('results/defence_plot.png', dpi=130, bbox_inches='tight')
print("\n✅ Plot saved to results/defence_plot.png")

# ── 8. Save results to text file ─────────────────────────────────────
with open('results/defence_results.txt', 'w') as f:
    f.write("DEFENCE EVALUATION RESULTS\n")
    f.write(f"Baseline (clean, no attack): {baseline:.1f}%\n\n")
    f.write(f"{'Defence':<25} {'FGSM':>8} {'PGD':>8} {'DeepFool':>10}\n")
    f.write("-"*55 + "\n")
    f.write(f"{'No defence (raw attack)':<25} {'22.4%':>8} {'0.0%':>8} {'0.0%':>10}\n")
    for i, name in enumerate(defence_names):
        f.write(f"{name:<25} {fgsm_accs[i]:>7.1f}% "
                f"{pgd_accs[i]:>7.1f}% {df_accs[i]:>9.1f}%\n")

print("✅ Results saved to results/defence_results.txt")
print("\n✅ Phase 4 complete! Run Phase 5: python3 app.py")