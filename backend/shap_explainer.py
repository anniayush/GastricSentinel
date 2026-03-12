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
    """
    Generates a SHAP explanation heatmap for the predicted class.
    Correctly handles both base ResNet50 and FusionModel architectures.
    """
    if shap is None or cv2 is None:
        return None

    img_tensor = preprocess_image(image_path).to(device)
    # Using a zero-tensor as background; for better results, use a small batch of training data
    background = torch.zeros_like(img_tensor).to(device)

    # Check if we are using the FusionModel
    is_fusion = hasattr(model, "clinical_net")

    # Define a wrapper to ensure SHAP passes data through the correct forward path
    def model_wrapper(x):
        with torch.no_grad():
            if is_fusion and clinical is not None and genomic is not None:
                # For Fusion, we must pass through feature_extractor then the fusion head
                # Note: We use the existing tensors provided by the predict function
                from predict import feature_extractor
                feat = feature_extractor(x).view(x.size(0), -1)
                return model(feat, clinical, genomic)
            return model(x)

    # Initialize Explainer
    explainer = shap.DeepExplainer(model_wrapper, background)
    
    # img_tensor requires_grad for DeepExplainer (gradient-based)
    img_tensor.requires_grad = True
    shap_values = explainer.shap_values(img_tensor)

    # Get the predicted index to visualize the correct class map
    with torch.no_grad():
        output = model_wrapper(img_tensor)
        pred_idx = int(torch.argmax(output, dim=1).item())

    # Process SHAP values: list of arrays [classes][batch, channels, height, width]
    # We take the map for the predicted class
    shap_map = shap_values[pred_idx][0]
    
    # Collapse channels and normalize
    shap_map = np.mean(shap_map, axis=0)
    shap_map = shap_map - shap_map.min()
    shap_map = shap_map / (shap_map.max() + 1e-8)
    shap_map = cv2.resize(shap_map, (512, 512))

    # Create overlay
    from PIL import Image
    img = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))
    heatmap = cv2.applyColorMap(np.uint8(255 * shap_map), cv2.COLORMAP_PLASMA)
    overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

    # Save to static directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, "frontend", "static", "uploads")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"shap_{uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, overlay)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"/static/uploads/{filename}"

def get_class_shap_scores(model, image_path, label_map, clinical=None, genomic=None):
    """
    Returns global importance scores per class based on SHAP values.
    """
    if shap is None:
        return None

    img_tensor = preprocess_image(image_path).to(device)
    background = torch.zeros_like(img_tensor).to(device)
    
    is_fusion = hasattr(model, "clinical_net")

    def _forward(x):
        if is_fusion and clinical is not None and genomic is not None:
            from predict import feature_extractor
            f = feature_extractor(x).view(x.size(0), -1)
            return model(f, clinical, genomic)
        return model(x)

    try:
        img_tensor.requires_grad = True
        explainer = shap.DeepExplainer(_forward, background)
        shap_vals = explainer.shap_values(img_tensor)

        result = {}
        for i, cls in enumerate(label_map):
            # Mean absolute SHAP value represents feature importance for that class
            class_map = shap_vals[i][0]
            result[cls] = float(np.mean(np.abs(class_map)))
        
        total = sum(result.values()) + 1e-8
        return {k: round(v / total, 4) for k, v in result.items()}
    
    except Exception as e:
        print(f"SHAP Score Error: {e}")
        return None