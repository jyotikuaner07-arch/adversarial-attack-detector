import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2
import os

from train_classifier import VictimCNN
from train_detector import DetectorCNN

device = torch.device('cpu')

# ── Load models at startup ───────────────────────────────────────────
victim_standard = VictimCNN().to(device)
victim_standard.load_state_dict(torch.load('models/victim_cnn.pth', map_location=device))
victim_standard.eval()

# Alias for backwards compatibility
victim = victim_standard

victim_robust = None
robust_path = 'models/victim_cnn_at.pth'
if os.path.exists(robust_path):
    try:
        victim_robust = VictimCNN().to(device)
        victim_robust.load_state_dict(torch.load(robust_path, map_location=device))
        victim_robust.eval()
    except Exception as e:
        print(f"⚠️ Failed loading robust model from {robust_path}: {e}")
        victim_robust = None

detector = DetectorCNN().to(device)
detector.load_state_dict(torch.load('models/detector_cnn.pth', map_location=device))
detector.eval()

NORMALISE = transforms.Normalize((0.1307,), (0.3081,))
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Clean MNIST test dataset
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=False, transform=transforms.ToTensor())


# ── Utilities ─────────────────────────────────────────────────────────
def tensor_to_base64(tensor):
    """Convert [1,1,28,28] tensor → base64 PNG string for the UI."""
    img_np = np.clip(tensor.squeeze().cpu().detach().numpy(), 0.0, 1.0)
    img_np = (img_np * 255).astype(np.uint8)
    pil = Image.fromarray(img_np, mode='L').resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Defence functions ─────────────────────────────────────────────────
def jpeg_defence(tensor_img, quality=50):
    """JPEG compression — strips high-frequency adversarial noise."""
    img_np = (tensor_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert('L')
    result = transforms.ToTensor()(np.array(compressed))
    return result.unsqueeze(0)


def gaussian_defence(tensor_img, sigma=0.5):
    """
    Gaussian blur — suppresses small local adversarial pixel spikes
    while retaining broad handwritten-digit strokes.
    Kernel size matches sigma: k = 2 * round(3 * sigma) + 1.
    """
    k = 2 * round(3 * sigma) + 1
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    img_np = tensor_img.squeeze().cpu().detach().numpy()
    blurred = cv2.GaussianBlur(img_np, (k, k), sigmaX=sigma, sigmaY=sigma)
    blurred = np.clip(blurred, 0.0, 1.0)
    return torch.tensor(blurred).unsqueeze(0).unsqueeze(0).float()


def median_defence(tensor_img, ksize=3):
    img_np = (tensor_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    filtered = cv2.medianBlur(img_np, ksize)
    result = filtered.astype(np.float32) / 255.0
    return torch.tensor(result).unsqueeze(0).unsqueeze(0).float()


def bit_depth_defence(tensor_img, bits=4):
    steps = 2 ** bits - 1
    result = torch.round(tensor_img * steps) / steps
    return result.float()


def test_defences(tensor, victim_model=None):
    candidates = [
        (gaussian_defence(tensor, sigma=0.5), "Gaussian Smoothing (sigma=0.5)"),
        (median_defence(tensor, ksize=3), "Median Filter (3x3)"),
        (bit_depth_defence(tensor, bits=4), "Bit-Depth Reduction (4-bit)"),
        (jpeg_defence(tensor, quality=50), "JPEG Compression (q=50)"),
    ]
    results = []
    for defended_tensor, defence_name in candidates:
        predicted_class, confidence = classify(defended_tensor, victim_model=victim_model)
        results.append({
            'tensor': defended_tensor,
            'defence': defence_name,
            'prediction': predicted_class,
            'confidence': confidence,
        })
    return results


# ── Model helpers ─────────────────────────────────────────────────────
def classify(tensor, victim_model=None):
    """Run VictimCNN — returns (digit_str, confidence_float)."""
    m = victim_model if victim_model is not None else victim_standard
    norm = NORMALISE(tensor.squeeze(0)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = m(norm)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return MNIST_CLASSES[pred.item()], round(conf.item() * 100, 1)


def detect(tensor):
    """
    Run DetectorCNN — returns (is_adversarial_bool, confidence_float).
    Note: DetectorCNN was trained on unnormalized [0, 1] tensors
    (ToTensor() without Normalize in train_detector.py), so we do not normalize here.
    """
    with torch.no_grad():
        logits = detector(tensor.to(device))
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return pred.item() == 1, round(conf.item() * 100, 1)


def estimate_noise(tensor):
    img_np = tensor.squeeze().cpu().numpy()
    clean_proxy = cv2.GaussianBlur(img_np, (3, 3), 0.5)
    diff = np.abs(img_np - clean_proxy)
    noise_pct = round(float(diff.mean()) * 100, 2)
    max_noise = round(float(diff.max()) * 100, 2)

    nonzero = diff[diff > 0.01]
    if len(nonzero) > 0:
        uniformity = float(np.std(nonzero) / (np.mean(nonzero) + 1e-8))
    else:
        uniformity = 1.0

    coverage = float((diff > 0.01).mean())

    return noise_pct, max_noise, uniformity, coverage


def format_pretty_attack(raw_atk):
    if not raw_atk:
        return None
    s = str(raw_atk).strip()
    s_lower = s.lower()
    if 'fgsm' in s_lower:
        return "FGSM — Fast Gradient Sign Method"
    if 'deepfool' in s_lower:
        return "DeepFool — Minimal perturbation"
    if 'pgd' in s_lower:
        return "PGD — Projected Gradient Descent"
    return s


def assess_input_quality(tensor):
    img_np = tensor.squeeze().numpy()
    mean_val = float(img_np.mean())
    max_val = float(img_np.max())
    std_val = float(img_np.std())

    if max_val < 0.1:
        return 'suspicious'
    if mean_val > 0.6:
        return 'suspicious'
    if std_val < 0.05:
        return 'suspicious'
    return 'mnist'
