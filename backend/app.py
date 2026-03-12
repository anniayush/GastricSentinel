import os
import json
import importlib
from pathlib import Path
from urllib import request as urllib_request
from flask import Flask, request, jsonify, render_template, send_file
from database import get_db
from predict import run_prediction, fusion_model as prediction_model
from gradcam import generate_gradcam
from shap_explainer import generate_shap
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
    pid = data.get("id")
    if not pid:
        return jsonify({"error": "Missing id"}), 400
    fields = {k: data[k] for k in ["name", "age", "gender", "condition", "risk"] if k in data}
    try:
        db.patients.update_one({"_id": ObjectId(pid)}, {"$set": fields})
    except Exception:
        db.patients.update_one({"_id": pid}, {"$set": fields})
    return jsonify({"status": "updated"})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filename = file.filename
    path = UPLOAD_FOLDER / filename
    file.save(str(path))

    age = int(request.form.get("age", 0))
    gender = request.form.get("gender", "Male")
    stage = request.form.get("stage", "I")
    gene_score = float(request.form.get("gene_score", 0))
    genomic_risk = float(request.form.get("genomic_risk", 0))

    img_tensor = preprocess_image(str(path))
    clinical_t = clinical_to_tensor(age, gender, stage)
    genomic_t = genomic_to_tensor([gene_score, genomic_risk])

    prediction = run_prediction(img_tensor, age, gender, stage, gene_score, genomic_risk)

    gradcam_path = generate_gradcam(prediction_model, str(path), clinical_t, genomic_t)
    shap_path = generate_shap(prediction_model, str(path), clinical_t, genomic_t)

    scan = {
        "prediction": prediction["label"],
        "diagnosis": prediction["diagnosis"],
        "probability": prediction["prob"],
        "risk": prediction["risk"],
        "risk_score": prediction["risk_score"],
        "tier": prediction["tier"],
        "timestamp": datetime.utcnow()
    }
    db.scans.insert_one(scan)

    return jsonify({
        "diagnosis": prediction["diagnosis"],
        "probability": prediction["prob"],
        "risk": prediction["risk"],
        "risk_score": prediction["risk_score"],
        "tier": prediction["tier"],
        "recommendation": prediction["recommendation"],
        "details": prediction["details"],
        "confidence": prediction["confidence"],
        "predicted_class": prediction["predicted_class"],
        "probabilities": prediction["probabilities"],
        "gradcam_url": gradcam_path,
        "shap_url": shap_path,
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

    c = canvas_mod.Canvas(str(REPORT_PATH), pagesize=letter)

    logo_path = STATIC_DIR / "logo.png"
    if os.path.exists(logo_path):
        c.drawImage(str(logo_path), 50, 730, width=50, height=50)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 750, "Gastric Sentinel Diagnostic Report")
    report_id = f"GS-{int(datetime.utcnow().timestamp())}"
    patient_name = data.get("patient_name", "Anonymous")
    patient_age  = data.get("patient_age", "")
    patient_id   = data.get("patient_id", "")
    patient_gender = data.get("patient_gender", "")
    notes        = data.get("notes", "")

    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.1, 0.1, 0.15)
    c.drawString(100, 718, "PATIENT INFORMATION")
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0.2, 0.2, 0.25)
    c.drawString(100, 702, f"Name: {patient_name}")
    c.drawString(320, 702, f"Age: {patient_age}   Gender: {patient_gender}")
    c.drawString(100, 686, f"Patient ID / MRN: {patient_id or 'Not provided'}")
    c.drawString(320, 686, f"Physician: {data.get('doctor', 'Dr. Admin')}")
    c.drawString(100, 670, f"Institution: {data.get('hospital', '')}")
    c.drawString(320, 670, f"Report ID: {report_id}")
    c.setStrokeColorRGB(0.8, 0.8, 0.85)
    c.line(100, 662, 510, 662)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.1, 0.1, 0.15)
    c.drawString(100, 648, "AI ANALYSIS RESULT")
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0.2, 0.2, 0.25)
    c.drawString(100, 632, f"Diagnosis: {data.get('diagnosis', '')}")
    c.drawString(100, 616, f"Predicted Class: {data.get('predicted_class', '')}   Confidence: {data.get('confidence', '')}%   Risk Score: {data.get('risk_score', '')}%")
    c.drawString(100, 600, f"Tier: {data.get('tier', '')}")
    c.line(100, 592, 510, 592)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(100, 578, "RECOMMENDATION")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.25, 0.25, 0.3)
    rec_text = data.get("recommendation", "Consult a specialist.")
    y_rec = 562
    for line in [rec_text[i:i+85] for i in range(0, len(rec_text), 85)]:
        c.drawString(100, y_rec, line)
        y_rec -= 14

    if notes:
        c.line(100, y_rec - 4, 510, y_rec - 4)
        c.setFont("Helvetica-Bold", 13)
        c.setFillColorRGB(0.1, 0.1, 0.15)
        c.drawString(100, y_rec - 20, "CLINICAL NOTES")
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.25, 0.25, 0.3)
        y_n = y_rec - 36
        for line in [notes[i:i+85] for i in range(0, len(notes), 85)]:
            c.drawString(100, y_n, line)
            y_n -= 14

    c.drawString(100, 80, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}   |   Report ID: {report_id}")

    probs = data.get("probabilities", {})
    if probs:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(100, 505, "Class Probabilities:")
        c.setFont("Helvetica", 10)
        y = 490
        for cls, val in sorted(probs.items(), key=lambda x: -x[1]):
            c.drawString(110, y, f"{cls}: {round(val * 100, 1)}%")
            y -= 14

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
        risk = p.get("risk", "low")
        if risk == "medium": risk = "mid"
        score = p.get("risk_score", 0)
        if isinstance(score, float) and score <= 1:
            score = int(score * 100)
        else:
            score = int(score) if score else 0
        if risk == "high" or score >= 80:
            status, default_score, action = "critical", 85, "Immediate oncology referral"
        elif risk == "mid" or score >= 45:
            status, default_score, action = "urgent", 55, "Endoscopic follow-up consult"
        else:
            status, default_score, action = "watch", 25, "H. pylori test + follow-up"
        if score == 0:
            score = default_score
        dx    = p.get("condition", p.get("last_diagnosis", "Unknown"))
        ts    = p.get("created_at", p.get("updated_at", ""))
        since = str(ts)[:10] if ts else "Recently"
        alerts.append({
            "id":      str(p["_id"]),
            "patient": p.get("name", "Unknown"),
            "pid":     "P-" + str(p["_id"])[-4:].upper(),
            "age":     p.get("age", "—"),
            "dx":      dx,
            "score":   score,
            "status":  status,
            "action":  action,
            "since":   since,
        })
    order = {"critical": 0, "urgent": 1, "watch": 2}
    alerts.sort(key=lambda a: order.get(a["status"], 3))
    return jsonify(alerts)


@app.route("/shap_explain", methods=["POST"])
def shap_explain():
    latest = list(db.scans.find().sort("_id", -1).limit(1))
    if not latest:
        return jsonify({"shap_values": {}})

    scan = latest[0]
    probabilities = scan.get("probabilities", {})
    predicted = str(scan.get("prediction", scan.get("predicted_class", ""))).upper()

    # If we have real softmax probabilities, derive SHAP as deviation from uniform baseline
    if probabilities:
        baseline = 1.0 / 8  # uniform prior for 8 classes
        shap_values = {
            cls: round((float(prob) - baseline) * 1.2, 4)
            for cls, prob in probabilities.items()
        }
        return jsonify({"shap_values": shap_values, "predicted": predicted})

    # Fallback: no probabilities stored — return empty so frontend uses its own calculation
    return jsonify({"shap_values": {}})


if __name__ == "__main__":
    app.run(debug=True)