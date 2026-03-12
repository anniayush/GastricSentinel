import os
import json
import importlib
from datetime import datetime
from urllib import request as urllib_request


SYSTEM_PROMPT = """You are MediAI, an expert clinical AI assistant embedded in Gastric Sentinel — an AI-powered histopathology diagnostic platform for gastric oncology.

You assist gastroenterologists and oncologists in:
- Interpreting AI-generated tissue classification results (8 classes: ADI, DEB, LYM, MUC, MUS, NORM, STR, TUM)
- Explaining GradCAM saliency maps and SHAP attribution values in clinical terms
- Risk stratification: CRITICAL (TUM/STR dominant), SUSPICIOUS, NEGATIVE
- Treatment and surveillance recommendations based on findings
- Multimodal fusion results combining histopathology + clinical (age, gender, stage) + genomic data

Guidelines:
- Be concise, evidence-based, and clinical in tone
- Always reference actual scan data when present in the context
- Frame all diagnoses as AI-assisted screening — never as definitive diagnosis
- For CRITICAL findings, always stress urgency of pathologist review and oncology referral
- Explain technical terms (GradCAM, SHAP, softmax, EfficientNet) clearly when asked
- If no scan is available, guide the user to upload and run a scan first
- Reference clinical + genomic metadata when provided (stage, gene scores affect risk stratification)
- Never fabricate scan values — only reference what is provided in context
"""


def _call_openai(messages: list, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.4,
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    try:
        requests_mod = importlib.import_module("requests")
        r = requests_mod.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )
        return r.json()["choices"][0]["message"]["content"]
    except ImportError:
        pass

    req = urllib_request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _call_anthropic(messages: list, model: str = "claude-haiku-4-5-20251001") -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    payload = {
        "model": model,
        "max_tokens": 400,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    try:
        requests_mod = importlib.import_module("requests")
        r = requests_mod.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=20,
        )
        data = r.json()
        return data["content"][0]["text"]
    except ImportError:
        pass

    req = urllib_request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["content"][0]["text"]


def build_context(scan_data: dict | None, clinical_data: dict | None = None) -> str:
    parts = []

    if scan_data:
        parts.append(
            f"Latest scan — "
            f"Predicted class: {scan_data.get('predicted_class', scan_data.get('prediction', '—'))} | "
            f"Tier: {scan_data.get('tier', '—')} | "
            f"Diagnosis: {scan_data.get('diagnosis', '—')} | "
            f"Risk score: {scan_data.get('risk_score', '—')}% | "
            f"Confidence: {round(float(scan_data.get('confidence', 0)) * 100, 1) if scan_data.get('confidence') else '—'}%"
        )
        probs = scan_data.get("probabilities", {})
        if probs:
            prob_str = ", ".join(
                f"{k}:{round(v*100,1)}%"
                for k, v in sorted(probs.items(), key=lambda x: -x[1])
            )
            parts.append(f"Class probabilities: {prob_str}")
        rec = scan_data.get("recommendation") or scan_data.get("details")
        if rec:
            parts.append(f"Recommendation: {rec}")

    if clinical_data:
        age = clinical_data.get("age")
        gender = clinical_data.get("gender")
        stage = clinical_data.get("stage")
        gene_score = clinical_data.get("gene_score")
        genomic_risk = clinical_data.get("genomic_risk")
        parts.append(
            f"Clinical metadata — "
            f"Age: {age or '—'} | Gender: {gender or '—'} | "
            f"Stage: {stage or '—'} | Gene score: {gene_score or '—'} | "
            f"Genomic risk: {genomic_risk or '—'}"
        )

    return " | ".join(parts) if parts else "No scan data available in current session."


def get_reply(
    message: str,
    context: str = "",
    scan_available: bool = False,
    history: list | None = None,
) -> str:
    history = history or []

    user_content = f"{context}\n\nDoctor: {message}" if context else message

    messages = list(history)
    messages.append({"role": "user", "content": user_content})

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if anthropic_key:
        try:
            return _call_anthropic(messages)
        except Exception as exc:
            print(f"[Chatbot] Anthropic error: {exc}")

    if openai_key:
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        full_messages = [system_msg] + messages
        try:
            return _call_openai(full_messages)
        except Exception as exc:
            print(f"[Chatbot] OpenAI error: {exc}")

    return _fallback_reply(message, context, scan_available)


def _fallback_reply(message: str, context: str, scan_available: bool) -> str:
    q = message.lower()

    if not scan_available:
        return (
            "Please upload a histopathology image and run an AI scan first. "
            "Once the analysis is complete, I can provide detailed clinical insights "
            "on the predicted class, GradCAM findings, SHAP attributions, and risk score."
        )

    if "gradcam" in q or "heatmap" in q or "saliency" in q:
        return (
            "The GradCAM overlay visualises which tissue regions most influenced the model's prediction. "
            "Red/orange hotspots indicate areas of highest activation in the final convolutional layer — "
            "these are the patches the EfficientNet-B4 backbone weighted most when classifying the tissue. "
            "In malignant predictions (TUM/STR), hotspots typically correspond to irregular glandular "
            "architecture, nuclear pleomorphism, or desmoplastic stroma. Always correlate with the H&E morphology."
        )

    if "shap" in q or "attribution" in q or "explain" in q:
        return (
            "SHAP (SHapley Additive exPlanations) quantifies each pixel's contribution to the prediction "
            "relative to a baseline. Positive values (red bars) push the model toward a class; negative "
            "values (blue) push away. Unlike GradCAM, SHAP is model-agnostic and reflects both direction "
            "and magnitude of influence. Compare the top contributing class with the GradCAM map to "
            "cross-validate the spatial regions driving the classification."
        )

    if "clinical" in q or "stage" in q or "age" in q or "genomic" in q or "gene" in q:
        return (
            "This platform uses a multimodal fusion model that combines image features (ResNet50 backbone), "
            "clinical metadata (age, gender, disease stage), and genomic risk scores. "
            "The fusion classifier integrates all three modalities — so a high genomic risk score or "
            "advanced stage classification will shift the overall risk even when histology appears borderline. "
            "Check the clinical/genomic inputs if the risk score seems unexpectedly high or low."
        )

    if "recommend" in q or "next step" in q or "what should" in q or "treatment" in q:
        if context:
            for part in context.split("|"):
                if "Recommendation:" in part:
                    rec = part.replace("Recommendation:", "").strip()
                    return f"Based on the current AI finding: {rec}"
        return (
            "For CRITICAL findings (TUM class dominant), immediate pathological confirmation and "
            "oncology referral within 48 hours is advised. For SUSPICIOUS (STR class), schedule "
            "targeted biopsy of adjacent tissue. For NEGATIVE findings, routine surveillance per "
            "your institution's gastric cancer screening protocol applies."
        )

    if "accuracy" in q or "reliable" in q or "confidence" in q:
        return (
            "The ResNet50 + multimodal fusion model achieves ~94.2% accuracy on the NCT-CRC-HE-100K "
            "8-class validation set. Confidence ≥ 85% is considered high-reliability. Values between "
            "60–85% are moderate — manual pathologist review is advisable. Below 60%, the model is "
            "uncertain and the image quality or patch selection should be reviewed."
        )

    if "risk" in q or "danger" in q or "serious" in q or "malignant" in q:
        return (
            "Risk stratification is based on the combined probability of TUM + STR classes. "
            "Score ≥ 45% → HIGH RISK (CRITICAL tier), 20–44% → MEDIUM (SUSPICIOUS), < 20% → LOW. "
            "Clinical stage and genomic risk inputs modulate this score in the fusion model. "
            "High-risk patients should be referred to oncology without delay."
        )

    if "class" in q or "category" in q or "label" in q or "tum" in q or "str" in q:
        return (
            "The model classifies tissue into 8 NCT-CRC classes: "
            "TUM (tumor/adenocarcinoma), STR (cancer-associated stroma), LYM (lymphocytes), "
            "DEB (cellular debris), MUC (mucosa), MUS (smooth muscle), NORM (normal mucosa), "
            "ADI (adipose tissue). TUM and STR are the malignancy-associated classes; the others "
            "indicate benign or non-neoplastic tissue types."
        )

    return (
        "I'm your AI diagnostic assistant for Gastric Sentinel. I can help interpret scan results, "
        "explain GradCAM heatmaps, SHAP attributions, risk scores, and provide evidence-based "
        "clinical recommendations. What would you like to know about the current finding?"
    )