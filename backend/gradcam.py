import torch
import numpy as np
from PIL import Image
import os
from uuid import uuid4
import threading
from utils import preprocess_image

try:
    import cv2
except ImportError:
    cv2 = None

gradcam_lock = threading.Lock()


def generate_gradcam(model, image, clinical=None, genomic=None):
    if cv2 is None:
        return None

    with gradcam_lock:
        model.eval()

        gradients = []
        activations = []

        target_layer = None
        if hasattr(model, "layer4"):
            target_layer = model.layer4
        elif hasattr(model, "features"):
            target_layer = model.features[-1]
        else:
            for submodule in model.modules():
                if hasattr(submodule, "layer4"):
                    target_layer = submodule.layer4
                    break
        if target_layer is None:
            target_layer = list(model.children())[-1]

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        img_tensor = preprocess_image(image).requires_grad_(True)

        is_fusion = hasattr(model, "clinical_net")

        if is_fusion and clinical is not None and genomic is not None:
            from model_loader import load_feature_extractor
            _feat_ext = load_feature_extractor()
            _feat_ext.eval()
            feat = _feat_ext(img_tensor).view(img_tensor.size(0), -1)
            output = model(feat, clinical, genomic)
        else:
            output = model(img_tensor)

        pred_class = int(torch.argmax(output, dim=1).item())
        loss = output[0, pred_class]

        model.zero_grad()
        loss.backward()

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        img = np.array(Image.open(image).convert("RGB").resize((512, 512)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        save_dir = os.path.join(base_dir, "frontend", "static", "uploads")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"gradcam_{uuid4().hex}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, overlay)

        forward_handle.remove()
        backward_handle.remove()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f"/static/uploads/{filename}"