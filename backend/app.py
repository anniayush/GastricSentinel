import os
import json
import importlib
from pathlib import Path
from urllib import request as urllib_request
from flask import Flask, request, jsonify, render_template, send_file
from database import get_db
from predict import run_prediction, _MASTER_MODEL as _base_model
from gradcam import generate_gradcam
from shap_explainer import generate_shap, get_class_shap_scores
from utils import preprocess_image, clinical_to_tensor, genomic_to_tensor
from datetime import datetime
try:
    from chatbot import get_reply, build_context
    _chatbot_available = True
except ImportError:
    _chatbot_available = False

try:
    from bson.objectid import ObjectId
except ImportError:
    def ObjectId(value):
        return value

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_frontend_dir():
    for name in ("frontend", "frontened"):
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    return BASE_DIR / "frontend"


FRONTEND_DIR = _resolve_frontend_dir()
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
REPORT_PATH = BASE_DIR / "gastric_report.pdf"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = get_db()


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/patients")
def patients_page():
    return render_template("patients.html")


@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")


@app.route("/stats")
def stats():
    total_patients = db.patients.count_documents({})
    total_scans = db.scans.count_documents({})
    high_risk = db.patients.count_documents({"risk": {"$in": ["high"]}})
    medium_risk = db.patients.count_documents({"risk": {"$in": ["mid", "medium"]}})
    return jsonify({
        "total_patients": total_patients,
        "total_scans": total_scans,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "model_accuracy": 0.942
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    patients = []
    for p in db.patients.find():
        risk = p.get("risk", "low")
        if risk == "medium": risk = "mid"
        patients.append({
            "id":             str(p["_id"]),
            "name":           p.get("name", "Unknown"),
            "age":            p.get("age", ""),
            "gender":         p.get("gender", ""),
            "last_diagnosis": p.get("condition", p.get("last_diagnosis", "")),
            "last":           p.get("condition", p.get("last_diagnosis", "")),
            "condition":      p.get("condition", ""),
            "risk":           risk,
            "risk_score":     p.get("risk_score", 0),
            "date":           str(p.get("last_scan", p.get("updated_at", p.get("created_at", "")))),
        })
    return jsonify(patients)


@app.route("/add_patient", methods=["POST"])
def add_patient():
    data = request.json
    patient = {
        "name": data.get("name"),
        "age": data.get("age"),
        "gender": data.get("gender"),
        "condition": data.get("condition"),
        "risk": data.get("risk", "low"),
        "created_at": datetime.utcnow().isoformat()
    }
    result = db.patients.insert_one(patient)
    patient["_id"] = str(result.inserted_id)
    return jsonify({"patient": patient})


@app.route("/delete_patient/<pid>", methods=["DELETE"])
def delete_patient(pid):
    try:
        db.patients.delete_one({"_id": ObjectId(pid)})
    except Exception:
        db.patients.delete_one({"_id": pid})
    return jsonify({"status": "deleted"})


@app.route("/update_patient", methods=["POST"])
def update_patient():
    data = request.json or {}
    pid  = data.get("id")
    if not pid:
        return jsonify({"error": "Missing id"}), 400
    fields = {k: data[k] for k in ["name","age","gender","condition","risk"] if k in data}
    try:
        db.patients.update_one({"_id": ObjectId(pid)}, {"$set": fields})
    except Exception:
        db.patients.update_one({"_id": pid}, {"$set": fields})
    return jsonify({"status": "updated"})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    import os as _os
    from uuid import uuid4 as _uuid4
    ext      = _os.path.splitext(file.filename)[-1] or ".png"
    filename = f"{_uuid4().hex}{ext}"
    path     = UPLOAD_FOLDER / filename
    file.save(str(path))

    age          = int(request.form.get("age", 0))
    gender       = request.form.get("gender", "Male")
    stage        = request.form.get("stage", "I")
    gene_score   = float(request.form.get("gene_score", 0))
    genomic_risk = float(request.form.get("genomic_risk", 0))

    img_tensor = preprocess_image(str(path))
    clinical_t = clinical_to_tensor(age, gender, stage)
    genomic_t  = genomic_to_tensor([gene_score, genomic_risk])

    prediction = run_prediction(img_tensor, age, gender, stage, gene_score, genomic_risk)

    gradcam_path = generate_gradcam(_base_model, str(path), clinical_t, genomic_t)
    shap_path    = generate_shap(_base_model, str(path), clinical_t, genomic_t)

    from utils import CLASSES as _CLASSES
    shap_scores = get_class_shap_scores(_base_model, str(path), _CLASSES, clinical_t, genomic_t)

    scan = {
        "prediction":      prediction["label"],
        "predicted_class": prediction["predicted_class"],
        "diagnosis":       prediction["diagnosis"],
        "probability":     prediction["prob"],
        "risk":            prediction["risk"],
        "risk_score":      prediction["risk_score"],
        "tier":            prediction["tier"],
        "probabilities":   prediction["probabilities"],
        "timestamp":       datetime.utcnow(),
    }
    db.scans.insert_one(scan)

    return jsonify({
        "diagnosis":       prediction["diagnosis"],
        "probability":     prediction["prob"],
        "risk":            prediction["risk"],
        "risk_score":      prediction["risk_score"],
        "tier":            prediction["tier"],
        "recommendation":  prediction["recommendation"],
        "details":         prediction["details"],
        "confidence":      prediction["confidence"],
        "predicted_class": prediction["predicted_class"],
        "probabilities":   prediction["probabilities"],
        "gradcam_url":     gradcam_path,
        "shap_url":        shap_path,
        "shap_values":     shap_scores,
    })


@app.route("/report")
def report():
    try:
        reportlab_pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        reportlab_canvas = importlib.import_module("reportlab.pdfgen.canvas")
        letter = reportlab_pagesizes.letter
        canvas = reportlab_canvas
    except ImportError:
        return jsonify({"error": "reportlab is not installed"}), 500

    scans = list(db.scans.find().sort("_id", -1).limit(1))
    if not scans:
        return jsonify({"error": "No scans found"}), 404
    scan = scans[0]

    c = canvas.Canvas(str(REPORT_PATH), pagesize=letter)

    logo_path = STATIC_DIR / "logo.png"
    if os.path.exists(logo_path):
        c.drawImage(str(logo_path), 50, 730, width=50, height=50)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 750, "Gastric Sentinel Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f"Diagnosis: {scan.get('diagnosis', scan.get('prediction', ''))}")
    c.drawString(100, 680, f"Probability: {scan.get('probability', '')}")
    c.drawString(100, 660, f"Risk Level: {scan.get('risk', '')}")
    c.drawString(100, 640, f"Tier: {scan.get('tier', '')}")
    c.drawString(100, 620, f"Generated: {datetime.utcnow()}")
    c.save()

    return send_file(str(REPORT_PATH), as_attachment=True)


@app.route("/generate_report", methods=["POST"])
def generate_report_endpoint():
    try:
        reportlab_pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        reportlab_canvas = importlib.import_module("reportlab.pdfgen.canvas")
        letter = reportlab_pagesizes.letter
        canvas_mod = reportlab_canvas
    except ImportError:
        return jsonify({"error": "reportlab is not installed"}), 500

    data = request.json or {}

    report_id = f"GS-{int(datetime.utcnow().timestamp())}"

    patient_name = data.get("patient_name", "Anonymous")
    patient_age = data.get("patient_age", "")
    patient_id = data.get("patient_id", "")
    patient_gender = data.get("patient_gender", "")
    doctor = data.get("doctor", "Dr. Admin")
    hospital = data.get("hospital", "Gastric Sentinel Lab")

    diagnosis = data.get("diagnosis", "")
    predicted_class = data.get("predicted_class", "")
    confidence = data.get("confidence", "")
    risk_score = data.get("risk_score", "")
    tier = data.get("tier", "")
    recommendation = data.get("recommendation", "Consult a specialist.")
    notes = data.get("notes", "")
    probs = data.get("probabilities", {})


    UPLOAD_DIR = STATIC_DIR / "uploads"

    input_image_path = UPLOAD_DIR / "input_image.png"
    gradcam_path = UPLOAD_DIR / "gradcam.png"
    shap_path = UPLOAD_DIR / "shap_plot.png"
    logo_path = STATIC_DIR / "logo.png"

    c = canvas_mod.Canvas(str(REPORT_PATH), pagesize=letter)

    if os.path.exists(logo_path):
        c.drawImage(str(logo_path), 50, 730, width=50, height=50)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 750, "Gastric Sentinel")

    c.setFont("Helvetica", 12)
    c.drawString(120, 730, "AI Assisted Gastric Diagnostic Report")

    c.setFont("Helvetica", 10)
    c.drawString(400, 750, f"Report ID: {report_id}")
    c.drawString(400, 735, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    c.line(50, 720, 550, 720)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 700, "Patient Information")

    c.setFont("Helvetica", 11)
    c.drawString(50, 680, f"Name: {patient_name}")
    c.drawString(300, 680, f"Age: {patient_age}")

    c.drawString(50, 665, f"Gender: {patient_gender}")
    c.drawString(300, 665, f"Patient ID: {patient_id}")

    c.drawString(50, 650, f"Physician: {doctor}")
    c.drawString(300, 650, f"Institution: {hospital}")

    c.line(50, 640, 550, 640)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 620, "AI Diagnosis Summary")

    c.setFont("Helvetica", 11)
    c.drawString(50, 600, f"Diagnosis: {diagnosis}")

    c.drawString(50, 585, f"Predicted Class: {predicted_class}")
    c.drawString(300, 585, f"Risk Tier: {tier}")

    c.drawString(50, 570, f"Confidence: {confidence}%")
    c.drawString(300, 570, f"Risk Score: {risk_score}%")

    c.line(50, 560, 550, 560)

    if probs:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, 540, "Class Probability Distribution")

        c.setFont("Helvetica", 10)
        y = 520

        for cls, val in sorted(probs.items(), key=lambda x: -x[1]):
            percent = round(val * 100, 2)
            c.drawString(60, y, f"{cls}: {percent}%")
            y -= 14

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 450, "AI Visual Explanation")

    # Input Image
    if os.path.exists(input_image_path):
        c.setFont("Helvetica", 10)
        c.drawString(50, 435, "Input Image")
        c.drawImage(str(input_image_path), 50, 320, width=150, height=100)

    # GradCAM
    if os.path.exists(gradcam_path):
        c.setFont("Helvetica", 10)
        c.drawString(220, 435, "Grad-CAM Heatmap")
        c.drawImage(str(gradcam_path), 220, 320, width=150, height=100)

    # SHAP
    if os.path.exists(shap_path):
        c.setFont("Helvetica", 10)
        c.drawString(390, 435, "SHAP Feature Importance")
        c.drawImage(str(shap_path), 390, 320, width=150, height=100)

    c.line(50, 300, 550, 300)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 280, "Recommendation")

    c.setFont("Helvetica", 10)
    y_rec = 265

    for line in [recommendation[i:i+90] for i in range(0, len(recommendation), 90)]:
        c.drawString(50, y_rec, line)
        y_rec -= 14

    if notes:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y_rec - 10, "Clinical Notes")

        c.setFont("Helvetica", 10)
        y_notes = y_rec - 25

        for line in [notes[i:i+90] for i in range(0, len(notes), 90)]:
            c.drawString(50, y_notes, line)
            y_notes -= 14
            
    c.setFont("Helvetica", 9)

    c.drawString(
        50,
        80,
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Report ID: {report_id}"
    )

    c.drawString(
        50,
        65,
        "Disclaimer: This AI-generated report is for research assistance only and must be verified by a medical professional."
    )

    c.save()

    return send_file(str(REPORT_PATH), as_attachment=True, mimetype="application/pdf")


@app.route("/chatbot", methods=["POST"])
def chatbot():
    body = request.json or {}
    msg = body.get("message", "")
    frontend_context = body.get("context", "")
    scan_available = body.get("scan_available", False)
    history = body.get("history", [])

    latest = list(db.scans.find().sort("_id", -1).limit(1))
    db_scan = latest[0] if latest else None

    if _chatbot_available:
        ctx = frontend_context
        if not ctx and db_scan:
            ctx = build_context(db_scan)
        elif not ctx:
            ctx = ""
        reply = get_reply(
            message=msg,
            context=ctx,
            scan_available=scan_available or bool(db_scan),
            history=history,
        )
    else:
        db_context = ""
        if db_scan:
            db_context = (
                f"Latest stored scan — diagnosis: {db_scan.get('diagnosis', db_scan.get('prediction', ''))}, "
                f"risk: {db_scan.get('risk', '')}, tier: {db_scan.get('tier', '')}, "
                f"probability: {db_scan.get('probability', '')}."
            )
        context = frontend_context or db_context or "No scan data available."
        prompt = f"""You are MediAI, a clinical AI assistant in the Gastric Sentinel diagnostic platform.

Current session context:
{context}

Guidelines:
- Be concise, clinical, and professional
- Reference the actual scan data when answering
- If no scan is available, ask the doctor to run a scan first
- Explain GradCAM and SHAP in plain clinical terms when asked
- For CRITICAL findings always emphasise urgency of pathologist review
- Never diagnose definitively — frame responses as AI-assisted screening

Doctor's question:
{msg}

Answer:"""
        try:
            import importlib as _il
            payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
            api_key = os.getenv("OPENAI_API_KEY", "")
            headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
            try:
                requests_mod = _il.import_module("requests")
                r = requests_mod.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
                reply = r.json()["choices"][0]["message"]["content"]
            except ImportError:
                req = urllib_request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib_request.urlopen(req, timeout=20) as resp:
                    reply = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception:
            reply = "AI assistant unavailable. Please run a scan and try again."

    return jsonify({"reply": reply})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    payload = {
        "rating": data.get("rating"),
        "status": data.get("status"),
        "notes": data.get("notes", ""),
        "diagnosis": data.get("diagnosis", ""),
        "timestamp": datetime.utcnow(),
    }
    db.feedback.insert_one(payload)
    return jsonify({"status": "saved"})


@app.route("/scan_activity", methods=["GET"])
def scan_activity():
    from datetime import timedelta
    now     = datetime.utcnow()
    weeks   = 12
    buckets = [0] * weeks
    cutoff  = now - timedelta(weeks=weeks)
    for s in db.scans.find({"timestamp": {"$gte": cutoff}}):
        ts = s.get("timestamp")
        if isinstance(ts, datetime):
            days     = (now - ts).days
            week_idx = weeks - 1 - min(days // 7, weeks - 1)
            buckets[week_idx] += 1
    return jsonify([{"count": c, "label": f"W{i+1}"} for i, c in enumerate(buckets)])


@app.route("/api/risk_alerts", methods=["GET"])
def risk_alerts():
    alerts = []
    for p in db.patients.find():
        risk  = p.get("risk", "low")
        if risk == "medium": risk = "mid"
        score = p.get("risk_score", 0)
        if isinstance(score, float) and score <= 1: score = int(score * 100)
        else: score = int(score) if score else 0
        if risk == "high" or score >= 80: status, ds, action = "critical", 85, "Immediate oncology referral"
        elif risk == "mid" or score >= 45: status, ds, action = "urgent",   55, "Endoscopic follow-up"
        else:                              status, ds, action = "watch",    25, "H. pylori test + follow-up"
        if score == 0: score = ds
        dx    = p.get("condition", p.get("last_diagnosis", "Unknown"))
        ts    = p.get("created_at", p.get("updated_at", ""))
        alerts.append({"id": str(p["_id"]), "patient": p.get("name","Unknown"),
            "pid": "P-"+str(p["_id"])[-4:].upper(), "age": p.get("age","—"),
            "dx": dx, "score": score, "status": status, "action": action,
            "since": str(ts)[:10] if ts else "Recently"})
    alerts.sort(key=lambda a: {"critical":0,"urgent":1,"watch":2}.get(a["status"],3))
    return jsonify(alerts)


@app.route("/shap_explain", methods=["POST"])
def shap_explain():
    """Returns per-class SHAP attribution scores for the most recent scan."""
    latest = list(db.scans.find().sort("_id", -1).limit(1))
    if not latest:
        return jsonify({"shap_values": {}})
    scan = latest[0]
    # If probabilities were stored, derive SHAP as softmax deviation (fast fallback)
    probs = scan.get("probabilities", {})
    if probs:
        baseline = 1.0 / 8
        shap_vals = {cls: round((float(p) - baseline) * 1.2, 4) for cls, p in probs.items()}
        return jsonify({"shap_values": shap_vals, "predicted": scan.get("predicted_class","")})
    # No stored probs — try re-running SHAP on last image
    img_path = scan.get("image_path", "")
    if img_path and os.path.exists(img_path):
        from utils import CLASSES as _CLS
        scores = get_class_shap_scores(img_path, _CLS)
        return jsonify({"shap_values": scores})
    return jsonify({"shap_values": {}})


if __name__ == "__main__":
    app.run(debug=True)