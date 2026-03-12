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
        # GradCAM must run on the CNN backbone (ResNet50 layer4), NOT the FusionModel.
        # If a FusionModel is passed, load the base ResNet50 separately for GradCAM.
        from model_loader import load_model
        cnn_model = load_model()
        cnn_model.eval()

        gradients  = []
        activations = []

        # layer4 always exists on ResNet50
        target_layer = cnn_model.layer4

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle  = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        img_tensor = preprocess_image(image).requires_grad_(True)

        # Always run GradCAM through the base ResNet50
        output = cnn_model(img_tensor)

        pred_class = int(torch.argmax(output, dim=1).item())
        loss = output[0, pred_class]

        cnn_model.zero_grad()
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