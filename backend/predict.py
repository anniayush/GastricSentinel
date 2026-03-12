import torch
import torch.nn.functional as F
from model_loader import load_model, load_feature_extractor, load_fusion_model
from utils import clinical_to_tensor, genomic_to_tensor, generate_report, CLASSES

model = load_model()
feature_extractor = load_feature_extractor()
fusion_model = load_fusion_model()

LABEL_MAP = CLASSES

TUM_IDX = LABEL_MAP.index("TUM")
STR_IDX = LABEL_MAP.index("STR")


def run_prediction(image_tensor, age, gender, stage, gene_score, genomic_risk):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_tensor = image_tensor.to(device)
    model.to(device)
    feature_extractor.to(device)
    fusion_model.to(device)

    model.eval()
    feature_extractor.eval()
    fusion_model.eval()

    clinical = clinical_to_tensor(age, gender, stage).to(device)
    genomic = genomic_to_tensor([gene_score, genomic_risk]).to(device)

    with torch.no_grad():
        img_features = feature_extractor(image_tensor)
        img_features = img_features.view(img_features.size(0), -1)

        fusion_logits = fusion_model(img_features, clinical, genomic)
        fusion_probs = F.softmax(fusion_logits, dim=1)

        pred_idx = int(torch.argmax(fusion_probs, dim=1).item())
        pred_confidence = float(fusion_probs[0, pred_idx].item())

    label = LABEL_MAP[pred_idx]

    tum_prob = float(fusion_probs[0, TUM_IDX].item())
    str_prob = float(fusion_probs[0, STR_IDX].item())
    cancer_risk_score = tum_prob + str_prob

    if cancer_risk_score >= 0.45:
        risk = "high"
    elif cancer_risk_score >= 0.20:
        risk = "medium"
    else:
        risk = "low"

    probabilities = {LABEL_MAP[i]: float(fusion_probs[0, i].item()) for i in range(len(LABEL_MAP))}

    report = generate_report(pred_idx, pred_confidence)

    return {
        "label": label,
        "prob": pred_confidence,
        "risk": risk,
        "risk_score": round(cancer_risk_score * 100),
        "probabilities": probabilities,
        "predicted_class": label,
        "diagnosis": report["diagnosis"],
        "tier": report["tier"],
        "recommendation": report["recommendation"],
        "details": report["details"],
        "confidence": pred_confidence,
    }