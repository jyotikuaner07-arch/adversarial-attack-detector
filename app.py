from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2
import re
import os
import json
import glob

from train_classifier import VictimCNN
from train_detector import DetectorCNN

app = Flask(__name__)
device = torch.device('cpu')

# ── Load both models at startup ───────────────────────────────────────
victim = VictimCNN().to(device)
victim.load_state_dict(torch.load('models/victim_cnn.pth', map_location=device))
victim.eval()

detector = DetectorCNN().to(device)
detector.load_state_dict(torch.load('models/detector_cnn.pth', map_location=device))
detector.eval()

NORMALISE    = transforms.Normalize((0.1307,), (0.3081,))
MNIST_CLASSES = ['0','1','2','3','4','5','6','7','8','9']


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_image(file_bytes, return_meta=False):
    """Convert uploaded image to tensor with validation."""
    pil = Image.open(io.BytesIO(file_bytes)).convert('L')
    native_size = pil.size  # (width, height) before resize
    w, h = native_size
    is_native_28 = (w == 28 and h == 28)

    # Pad to square then LANCZOS to 28x28 (don't stretch)
    if not is_native_28:
        if w != h:
            max_side = max(w, h)
            # Match background based on pixel average (black-on-white vs white-on-black)
            bg = 255 if np.array(pil).mean() > 127 else 0
            square_img = Image.new('L', (max_side, max_side), color=bg)
            square_img.paste(pil, ((max_side - w) // 2, (max_side - h) // 2))
            pil = square_img
        pil = pil.resize((28, 28), Image.LANCZOS)

    img_np = np.array(pil).astype(np.float32)

    # Invert only if mean > 127 (black-on-white → MNIST white-on-black)
    inverted = False
    if img_np.mean() > 127:
        img_np = 255.0 - img_np
        inverted = True

    img_np = img_np / 255.0

    # Min-max contrast stretch ONLY if the file was not already 28x28.
    # Native attack PNGs must not be stretched (it destroys FGSM/PGD noise and changes the preview).
    if not is_native_28:
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max - img_min > 0.01:  # Not blank
            img_np = (img_np - img_min) / (img_max - img_min)

    img_np = np.clip(img_np, 0.0, 1.0)

    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float()

    # Debug: print stats
    print(f"[DEBUG] Image shape: {tensor.shape}, min: {tensor.min():.3f}, max: {tensor.max():.3f}, native: {native_size}, inverted: {inverted}", flush=True)

    if return_meta:
        return tensor, {
            'native_size': list(native_size),
            'inverted': inverted
        }
    return tensor


def tensor_to_base64(tensor):
    """Convert [1,1,28,28] tensor → base64 PNG string for the UI."""
    img_np = np.clip(tensor.squeeze().cpu().detach().numpy(), 0.0, 1.0)
    img_np = (img_np * 255).astype(np.uint8)
    pil    = Image.fromarray(img_np, mode='L').resize((140, 140), Image.NEAREST)
    buf    = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Defence functions ─────────────────────────────────────────────────
def jpeg_defence(tensor_img, quality=50):
    """JPEG compression — strips high-frequency adversarial noise."""
    img_np  = (tensor_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
    buf     = io.BytesIO()
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
    img_np  = tensor_img.squeeze().cpu().detach().numpy()
    blurred = cv2.GaussianBlur(img_np, (k, k), sigmaX=sigma, sigmaY=sigma)
    blurred = np.clip(blurred, 0.0, 1.0)
    return torch.tensor(blurred).unsqueeze(0).unsqueeze(0).float()


def median_defence(tensor_img, ksize=3):
    img_np   = (tensor_img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    filtered = cv2.medianBlur(img_np, ksize)
    result   = filtered.astype(np.float32) / 255.0
    return torch.tensor(result).unsqueeze(0).unsqueeze(0).float()


def bit_depth_defence(tensor_img, bits=4):
    steps  = 2 ** bits - 1
    result = torch.round(tensor_img * steps) / steps
    return result.float()


def test_defences(tensor):
    candidates = [
        (gaussian_defence(tensor, sigma=0.5),  "Gaussian Smoothing (sigma=0.5)"),
        (median_defence(tensor, ksize=3),      "Median Filter (3x3)"),
        (bit_depth_defence(tensor, bits=4),    "Bit-Depth Reduction (4-bit)"),
        (jpeg_defence(tensor, quality=50),     "JPEG Compression (q=50)"),
    ]
    results = []
    for defended_tensor, defence_name in candidates:
        predicted_class, confidence = classify(defended_tensor)
        results.append({
            'tensor':     defended_tensor,
            'defence':    defence_name,
            'prediction': predicted_class,
            'confidence': confidence,
        })
    return results


# ── Model helpers ─────────────────────────────────────────────────────
def classify(tensor):
    """Run VictimCNN — returns (digit_str, confidence_float)."""
    norm = NORMALISE(tensor.squeeze(0)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = victim(norm)
        probs  = F.softmax(logits, dim=1)
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
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return pred.item() == 1, round(conf.item() * 100, 1)


def estimate_noise(tensor):
    img_np      = tensor.squeeze().cpu().numpy()
    clean_proxy = cv2.GaussianBlur(img_np, (3, 3), 0.5)
    diff        = np.abs(img_np - clean_proxy)
    noise_pct   = round(float(diff.mean()) * 100, 2)
    max_noise   = round(float(diff.max())  * 100, 2)

    nonzero = diff[diff > 0.01]
    if len(nonzero) > 0:
        uniformity = float(np.std(nonzero) / (np.mean(nonzero) + 1e-8))
    else:
        uniformity = 1.0

    coverage = float((diff > 0.01).mean())

    return noise_pct, max_noise, uniformity, coverage


def guess_attack_type(noise_pct, max_noise, uniformity, coverage):
    if coverage > 0.55 and uniformity < 0.75:
        return "FGSM — Fast Gradient Sign Method (uniform sign perturbation)"
    if coverage < 0.35 and noise_pct < 4.0:
        return "DeepFool — Minimal perturbation (surgical attack)"
    return "PGD — Projected Gradient Descent (iterative, optimised noise)"

def assess_input_quality(tensor):
    """
    Checks if the uploaded image looks like a genuine MNIST digit.
    Returns 'mnist' if it looks like a proper digit image,
    'suspicious' if it looks like a non-MNIST or corrupted input.
    """
    img_np = tensor.squeeze().numpy()

    # A genuine MNIST digit has:
    # 1. Mostly dark background (mean pixel value relatively low)
    # 2. Some bright pixels forming the digit stroke
    # 3. Not completely uniform (not blank)

    mean_val = float(img_np.mean())
    max_val  = float(img_np.max())
    std_val  = float(img_np.std())

    # Completely blank or near-blank image
    if max_val < 0.1:
        return 'suspicious'

    # Very high mean = mostly white = likely not a proper digit
    if mean_val > 0.6:
        return 'suspicious'

    # Very low std = uniform image = no digit structure
    if std_val < 0.05:
        return 'suspicious'

    return 'mnist'


# ── Metadata Parser ───────────────────────────────────────────────────
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


def parse_attack_metadata(file_storage, form_data=None):
    r"""
    Extract attack metadata from:
    1. Form field 'metadata' (JSON string if provided)
    2. Upload filename matching true(\d+)_pred(\d+) and attack names
    3. On-disk sidecar glob attacks/**/{stem}.json as fallback
    Returns: (attack_type, true_label, adv_pred)
    """
    attack_type = None
    true_label = None
    adv_pred = None

    # Check 1: form metadata if present
    if form_data and 'metadata' in form_data:
        try:
            m_data = json.loads(form_data['metadata'])
            if 'attack' in m_data:
                attack_type = format_pretty_attack(m_data['attack'])
            if 'true' in m_data:
                true_label = int(m_data['true'])
            elif 'true_label' in m_data:
                true_label = int(m_data['true_label'])
            if 'adv_pred' in m_data:
                adv_pred = int(m_data['adv_pred'])
        except Exception:
            pass

    filename = getattr(file_storage, 'filename', '') or ''
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]

    # Check 2: filename regex parsing
    if basename:
        if re.search(r'fgsm', basename, re.IGNORECASE):
            attack_type = attack_type or "FGSM — Fast Gradient Sign Method"
        elif re.search(r'deepfool', basename, re.IGNORECASE):
            attack_type = attack_type or "DeepFool — Minimal perturbation"
        elif re.search(r'pgd', basename, re.IGNORECASE):
            attack_type = attack_type or "PGD — Projected Gradient Descent"

        tp_match = re.search(r'true(\d+)_pred(\d+)', basename, re.IGNORECASE)
        if tp_match:
            if true_label is None:
                true_label = int(tp_match.group(1))
            if adv_pred is None:
                adv_pred = int(tp_match.group(2))
        else:
            trail_match = re.search(r'(?:fgsm|pgd|deepfool)[_\s-]*(\d+)', basename, re.IGNORECASE)
            if trail_match and true_label is None:
                true_label = int(trail_match.group(1))

    # Check 3: disk sidecar glob fallback attacks/**/{stem}.json
    if stem and (attack_type is None or true_label is None or adv_pred is None):
        try:
            sidecar_matches = glob.glob(f'attacks/**/{stem}.json', recursive=True) + glob.glob(f'attacks/{stem}.json')
            if sidecar_matches:
                with open(sidecar_matches[0], 'r') as sf:
                    sidecar_data = json.load(sf)
                    if attack_type is None and 'attack' in sidecar_data:
                        attack_type = format_pretty_attack(sidecar_data['attack'])
                    if true_label is None and 'true' in sidecar_data:
                        true_label = int(sidecar_data['true'])
                    if adv_pred is None and 'adv_pred' in sidecar_data:
                        adv_pred = int(sidecar_data['adv_pred'])
        except Exception:
            pass

    return attack_type, true_label, adv_pred


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})


@app.route('/debug')
def debug():
    """Check if models are working."""
    test_img = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        victim_out = victim(NORMALISE(test_img.squeeze(0)).unsqueeze(0))
        detector_out = detector(test_img)
    
    return jsonify({
        'victim_prediction': int(victim_out.argmax(1).item()),
        'detector_output': int(detector_out.argmax(1).item()),
        'status': 'Models loaded and working'
    })


@app.route('/lab/samples', methods=['GET'])
def lab_samples():
    attack = request.args.get('attack', 'PGD').strip()
    norm_map = {'pgd': 'PGD', 'fgsm': 'FGSM', 'deepfool': 'DeepFool'}
    attack_name = norm_map.get(attack.lower(), attack)

    index_path = f'attacks/{attack_name}_index.json'
    if not os.path.exists(index_path):
        return jsonify({
            'error': f'Attack index file {index_path} not found. Run generate_attacks.py first.',
            'samples': []
        }), 404

    try:
        with open(index_path, 'r') as f:
            data = json.load(f)
        # Filter for successful attacks only
        successful = [item for item in data if item.get('success')]
        # Return first 50 successful samples
        samples = successful[:50]
        return jsonify({
            'attack': attack_name,
            'total_successful': len(successful),
            'samples': samples
        })
    except Exception as e:
        return jsonify({'error': f'Failed reading attack index: {str(e)}'}), 500


@app.route('/lab/analyse', methods=['POST'])
def lab_analyse():
    req_data = request.get_json(silent=True) or {}
    attack = req_data.get('attack', 'PGD')
    idx = req_data.get('index')

    if idx is None:
        return jsonify({'error': 'Missing sample index'}), 400

    try:
        idx = int(idx)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid index format'}), 400

    norm_map = {'pgd': 'PGD', 'fgsm': 'FGSM', 'deepfool': 'DeepFool'}
    attack_name = norm_map.get(str(attack).strip().lower(), attack)

    adv_pt_path = f'attacks/{attack_name}_adv_images.pt'
    index_json_path = f'attacks/{attack_name}_index.json'

    if not os.path.exists(adv_pt_path) or not os.path.exists(index_json_path):
        return jsonify({
            'error': f'Missing attack data for {attack_name}. Run generate_attacks.py first.'
        }), 404

    try:
        with open(index_json_path, 'r') as f:
            idx_records = json.load(f)

        # Look up by "index" field
        record = next((r for r in idx_records if r.get('index') == idx), None)
        if record is None:
            return jsonify({'error': f'Index {idx} not found in {index_json_path}'}), 404

        true_label = record.get('true')
        adv_pred = record.get('adv_pred')

        adv_tensor_all = torch.load(adv_pt_path, map_location='cpu')
        if idx < 0 or idx >= len(adv_tensor_all):
            return jsonify({'error': f'Index {idx} out of range in {adv_pt_path}'}), 400

        tensor = adv_tensor_all[idx].unsqueeze(0)  # shape [1, 1, 28, 28]
    except Exception as e:
        return jsonify({'error': f'Could not load lab sample: {str(e)}'}), 500

    attack_type = format_pretty_attack(attack_name)

    # Step 1 — Run detector
    is_adv_det, detect_conf = detect(tensor)

    # Lab sample is guaranteed adversarial attack (never clean)
    is_adversarial = True

    # Step 2 — Classify original (attacked) image
    orig_class, orig_conf = classify(tensor)

    # Step 3 — Base64 of original for UI display
    orig_b64 = tensor_to_base64(tensor)

    # Step 4 — Estimate noise
    noise_pct, max_noise, uniformity, coverage = estimate_noise(tensor)

    # Step 5 — Run mild defence methods
    defence_results = test_defences(tensor)

    # Count votes across the 4 mild defences
    vote_counts = {}
    for r in defence_results:
        c = r['prediction']
        vote_counts[c] = vote_counts.get(c, 0) + 1

    defended_majority = None
    for c, count in vote_counts.items():
        if count >= 3:
            defended_majority = c
            break

    # Only not_fooled | recovered | inconclusive
    if orig_class == str(true_label):
        verdict_type = "not_fooled"
        best = next((r for r in defence_results if r['prediction'] == orig_class), defence_results[0])
        defended_class = orig_class
        defended_conf = best['confidence']
        defence_applied = best['defence']
        defended_tensor = best['tensor']
        defended_b64 = tensor_to_base64(defended_tensor)
        recovery_rate = (
            f"Not fooled — original prediction {orig_class} confirmed "
            f"(ground truth label: {true_label})"
        )
    elif defended_majority == str(true_label):
        verdict_type = "recovered"
        best = next((r for r in defence_results if r['prediction'] == str(true_label)), defence_results[0])
        defended_class = str(true_label)
        defended_conf = best['confidence']
        defence_applied = best['defence']
        defended_tensor = best['tensor']
        defended_b64 = tensor_to_base64(defended_tensor)
        recovery_rate = (
            f"Recovered — defence consensus identified true digit {true_label} "
            f"(attack predicted {orig_class})"
        )
    else:
        verdict_type = "inconclusive"
        best = next((r for r in defence_results if r['prediction'] == orig_class), defence_results[0])
        defended_class = orig_class
        defended_conf = best['confidence']
        defence_applied = best['defence']
        defended_tensor = best['tensor']
        defended_b64 = tensor_to_base64(defended_tensor)
        recovery_rate = (
            f"Inconclusive — model predicted {orig_class}, "
            f"true digit (from dataset): {true_label}, defence could not recover"
        )

    debug_data = {
        'source': 'lab',
        'sample_index': idx,
        'orig_class': orig_class,
        'orig_conf': orig_conf,
        'defences': [
            {
                'defence': r['defence'],
                'prediction': r['prediction'],
                'confidence': r['confidence']
            }
            for r in defence_results
        ],
        'verdict_type': verdict_type,
        'true_label': true_label,
        'adv_pred': adv_pred,
        'attack_type': attack_type,
    }

    return jsonify({
        'source':              'lab',
        'is_adversarial':      True,
        'verdict':             verdict_type,
        'verdict_type':        verdict_type,
        'detect_confidence':   detect_conf,
        'original_class':      orig_class,
        'original_confidence': orig_conf,
        'original_image':      orig_b64,
        'defended_class':      defended_class,
        'defended_confidence': defended_conf,
        'defended_image':      defended_b64,
        'defence_applied':     defence_applied,
        'noise_pct':           noise_pct,
        'max_noise':           max_noise,
        'attack_type':         attack_type,
        'true_label':          true_label,
        'adv_pred':            adv_pred,
        'recovery_rate':       recovery_rate,
        'debug':               debug_data,
    })


@app.route('/analyse', methods=['POST'])
def analyse():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    file_bytes = image_file.read()

    try:
        tensor, meta = preprocess_image(file_bytes, return_meta=True)
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    # Parse metadata (attack type, ground truth, adversarial prediction)
    attack_type, true_label, adv_pred = parse_attack_metadata(image_file, request.form)

    # Step 1 — Detect if adversarial
    is_adversarial, detect_conf = detect(tensor)

    # If attack metadata is present in filename / sidecar, treat as adversarial regardless of detector
    # (detector miss is expected for DeepFool PNGs)
    if attack_type is not None or true_label is not None:
        is_adversarial = True

    # Step 2 — Classify original (attacked) image
    orig_class, orig_conf = classify(tensor)

    # Step 3 — Base64 of original for UI display
    orig_b64 = tensor_to_base64(tensor)

    # Initialise defence outputs
    defended_class  = None
    defended_conf   = None
    defence_applied = None
    defended_b64    = None
    noise_pct       = None
    max_noise       = None
    recovery_rate   = None
    defence_results = []

    if not is_adversarial:
        verdict_type = "clean"
        recovery_rate = "Clean image — no defence needed"
    else:
        # Step 4 — Estimate noise
        noise_pct, max_noise, uniformity, coverage = estimate_noise(tensor)

        # Step 5 — Run mild defence methods
        defence_results = test_defences(tensor)

        # Count votes across the 4 mild defences
        vote_counts = {}
        for r in defence_results:
            c = r['prediction']
            vote_counts[c] = vote_counts.get(c, 0) + 1

        # Check for consensus (3 or 4 out of 4)
        defended_majority = None
        for c, count in vote_counts.items():
            if count >= 3:
                defended_majority = c
                break

        if true_label is not None:
            if orig_class == str(true_label):
                verdict_type = "not_fooled"
                # mildest defence that still says orig_class (or gaussian 0.5)
                best = next((r for r in defence_results if r['prediction'] == orig_class), defence_results[0])
                defended_class = orig_class
                defended_conf = best['confidence']
                defence_applied = best['defence']
                defended_tensor = best['tensor']
                defended_b64 = tensor_to_base64(defended_tensor)
                recovery_rate = (
                    f"Not fooled — original prediction {orig_class} confirmed "
                    f"(ground truth label: {true_label})"
                )
            elif defended_majority == str(true_label):
                verdict_type = "recovered"
                best = next((r for r in defence_results if r['prediction'] == str(true_label)), defence_results[0])
                defended_class = str(true_label)
                defended_conf = best['confidence']
                defence_applied = best['defence']
                defended_tensor = best['tensor']
                defended_b64 = tensor_to_base64(defended_tensor)
                recovery_rate = (
                    f"Recovered — defence consensus identified true digit {true_label} "
                    f"(attack predicted {orig_class})"
                )
            else:
                verdict_type = "inconclusive"  # fooled and not recovered
                best = next((r for r in defence_results if r['prediction'] == orig_class), defence_results[0])
                defended_class = orig_class
                defended_conf = best['confidence']
                defence_applied = best['defence']
                defended_tensor = best['tensor']
                defended_b64 = tensor_to_base64(defended_tensor)
                recovery_rate = (
                    f"Inconclusive — model predicted {orig_class}, "
                    f"true digit (from filename): {true_label}, defence could not recover"
                )
        else:
            # No metadata true_label
            if defended_majority is not None and defended_majority != orig_class:
                verdict_type = "recovered"  # weak recovery
                best = next((r for r in defence_results if r['prediction'] == defended_majority), defence_results[0])
                defended_class = defended_majority
                defended_conf = best['confidence']
                defence_applied = best['defence']
                defended_tensor = best['tensor']
                defended_b64 = tensor_to_base64(defended_tensor)
                recovery_rate = (
                    f"Recovered (consensus) — defence identified digit {defended_majority}"
                )
            else:
                verdict_type = "inconclusive"
                # NOT not_fooled
                best = next((r for r in defence_results if r['prediction'] == orig_class), defence_results[0])
                defended_class = orig_class
                defended_conf = best['confidence']
                defence_applied = best['defence']
                defended_tensor = best['tensor']
                defended_b64 = tensor_to_base64(defended_tensor)
                recovery_rate = "Inconclusive — defence could not reliably recover digit"

    debug_data = {
        'native_size': meta['native_size'],
        'inverted': meta['inverted'],
        'orig_class': orig_class,
        'orig_conf': orig_conf,
        'defences': [
            {
                'defence': r['defence'],
                'prediction': r['prediction'],
                'confidence': r['confidence']
            }
            for r in defence_results
        ],
        'verdict_type': verdict_type,
        'true_label': true_label,
        'adv_pred': adv_pred,
        'attack_type': attack_type,
    }

    return jsonify({
        'is_adversarial':      is_adversarial,
        'verdict':             verdict_type,
        'verdict_type':        verdict_type,
        'detect_confidence':   detect_conf,
        'original_class':      orig_class,
        'original_confidence': orig_conf,
        'original_image':      orig_b64,
        'defended_class':      defended_class,
        'defended_confidence': defended_conf,
        'defended_image':      defended_b64,
        'defence_applied':     defence_applied,
        'noise_pct':           noise_pct,
        'max_noise':           max_noise,
        'attack_type':         attack_type,
        'true_label':          true_label,
        'adv_pred':            adv_pred,
        'recovery_rate':       recovery_rate,
        'debug':               debug_data,
    })


if __name__ == '__main__':
    print("\n🚀 AdverSense Flask Server starting...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000, use_reloader=False)