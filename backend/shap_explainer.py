import torch
import numpy as np
import os
from uuid import uuid4
from utils import preprocess_image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import shap
except ImportError:
    shap = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_shap(model, image_path, clinical=None, genomic=None):
    if shap is None or cv2 is None:
        return None

    img_tensor = preprocess_image(image_path).to(device).requires_grad_(True)
    background = torch.zeros_like(img_tensor)

    is_fusion = hasattr(model, "clinical_net")

    with torch.no_grad():
        if is_fusion and clinical is not None and genomic is not None:
            from model_loader import load_feature_extractor
            _feat_ext = load_feature_extractor().to(device)
            _feat_ext.eval()
            feat = _feat_ext(img_tensor).view(img_tensor.size(0), -1)
            out = model(feat, clinical, genomic)
        else:
            out = model(img_tensor)
        pred_idx = int(torch.argmax(out, dim=1).item())

    def model_forward(x):
        if is_fusion and clinical is not None and genomic is not None:
            from model_loader import load_feature_extractor
            _fe = load_feature_extractor().to(device)
            _fe.eval()
            f = _fe(x).view(x.size(0), -1)
            return model(f, clinical, genomic)
        return model(x)

    explainer = shap.DeepExplainer(model_forward, background)
    shap_values = explainer.shap_values(img_tensor)

    shap_map = shap_values[pred_idx][0]
    shap_map = np.mean(shap_map, axis=0)
    shap_map = shap_map - shap_map.min()
    shap_map = shap_map / (shap_map.max() + 1e-8)
    shap_map = cv2.resize(shap_map, (512, 512))

    from PIL import Image
    img = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))
    heatmap = cv2.applyColorMap(np.uint8(255 * shap_map), cv2.COLORMAP_PLASMA)
    overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, "frontend", "static", "uploads")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"shap_{uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, overlay)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"/static/uploads/{filename}"


def get_shap_values_per_class(model, image_path, label_map, clinical=None, genomic=None):
    if shap is None:
        return None

    img_tensor = preprocess_image(image_path).to(device).requires_grad_(True)
    background = torch.zeros_like(img_tensor)

    def model_forward(x):
        if clinical is not None and genomic is not None:
            return model(x, clinical, genomic)
        return model(x)

    explainer = shap.DeepExplainer(model_forward, background)
    shap_values = explainer.shap_values(img_tensor)

    result = {}
    for i, cls in enumerate(label_map):
        class_shap = shap_values[i][0]
        result[cls] = float(np.mean(np.abs(class_shap)))

    total = sum(result.values()) + 1e-8
    result = {k: round(v / total, 4) for k, v in result.items()}

    return result