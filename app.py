from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import os

# Import both models from their respective files
from train_classifier import VictimCNN
from train_detector import DetectorCNN

app = Flask(__name__)

# ── 1. Device & load both models ──────────────────────────────────────
device = torch.device('cpu')  # Flask runs on CPU

# Load VictimCNN
victim = VictimCNN().to(device)
victim.load_state_dict(torch.load('models/victim_cnn.pth',
                                   map_location=device))
victim.eval()
print("✅ VictimCNN loaded")

# Load DetectorCNN
detector = DetectorCNN().to(device)
detector.load_state_dict(torch.load('models/detector_cnn.pth',
                                     map_location=device))
detector.eval()
print("✅ DetectorCNN loaded")

# ── 2. Preprocessing ──────────────────────────────────────────────────
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))
MNIST_CLASSES = ['0','1','2','3','4','5','6','7','8','9']

def preprocess_image(file_bytes):
    """Convert uploaded image bytes to a 28x28 greyscale tensor."""
    pil = Image.open(io.BytesIO(file_bytes)).convert('L')  # greyscale
    pil = pil.resize((28, 28))                             # MNIST size
    tensor = transforms.ToTensor()(pil)                    # [1, 28, 28]
    return tensor.unsqueeze(0)                             # [1, 1, 28, 28]

def jpeg_defence(tensor_img, quality=75):
    """Strip adversarial noise using JPEG compression."""
    img_np = (tensor_img.squeeze().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    result = transforms.ToTensor()(np.array(compressed))
    return result.unsqueeze(0)

# ── 3. Prediction helpers ─────────────────────────────────────────────
def classify(tensor):
    """Run VictimCNN — returns predicted digit and confidence %."""
    norm = NORMALISE(tensor.squeeze(0)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = victim(norm)
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return MNIST_CLASSES[pred.item()], round(conf.item() * 100, 1)

def detect(tensor):
    """Run DetectorCNN — returns True if adversarial + confidence %."""
    with torch.no_grad():
        logits = detector(tensor.to(device))
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    is_adversarial = pred.item() == 1
    return is_adversarial, round(conf.item() * 100, 1)

# ── 4. Routes ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main demo page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Quick check that server is running."""
    return jsonify({'status': 'ok', 'models_loaded': True})

@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Main endpoint. Accepts image upload.
    Returns: classification, detection verdict, defence result.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file_bytes = request.files['image'].read()

    # Step 1: Convert image to tensor
    try:
        tensor = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    # Step 2: Detect if adversarial
    is_adversarial, detect_conf = detect(tensor)

    # Step 3: Classify original image
    orig_class, orig_conf = classify(tensor)

    # Step 4: Apply defence if adversarial, reclassify
    defended_class    = None
    defended_conf     = None
    defence_applied   = None

    if is_adversarial:
        defended_tensor  = jpeg_defence(tensor, quality=75)
        defended_class, defended_conf = classify(defended_tensor)
        defence_applied  = "JPEG Compression (quality=75)"

    return jsonify({
        'is_adversarial':     is_adversarial,
        'detect_confidence':  detect_conf,
        'original_class':     orig_class,
        'original_confidence': orig_conf,
        'defended_class':     defended_class,
        'defended_confidence': defended_conf,
        'defence_applied':    defence_applied,
    })

# ── 5. Run server ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 Starting AdverSense Flask Server...")
    print("   Open http://localhost:5000 in your browser")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000, use_reloader=False)