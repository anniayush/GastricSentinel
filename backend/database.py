import importlib
import os
from datetime import datetime

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://gastric_admin:Gastric12345@cluster0.9uel17u.mongodb.net/?appName=Cluster0"
)


class _InMemoryInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class _InMemoryCursor:
    """Tiny iterable cursor with Mongo-like sort/limit chaining."""
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        reverse = direction == -1
        self._docs.sort(key=lambda d: d.get(key), reverse=reverse)
        return self

    def limit(self, n):
        self._docs = self._docs[: max(int(n), 0)]
        return self

    def __iter__(self):
        return iter(self._docs)


class _InMemoryCollection:
    """
    Fallback in-memory MongoDB-compatible collection.
    Used when pymongo is not installed or MongoDB is unreachable.
    """
    def __init__(self):
        self._docs = []
        self._counter = 1

    def insert_one(self, doc):
        stored = dict(doc)
        stored["_id"] = str(self._counter)
        self._counter += 1
        self._docs.append(stored)
        return _InMemoryInsertResult(stored["_id"])

    def find(self, query=None, projection=None):
        query      = query      or {}
        projection = projection or {}
        out = []
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                d = dict(doc)
                if projection.get("_id") == 0:
                    d.pop("_id", None)
                out.append(d)
        return _InMemoryCursor(out)

    def find_one(self, query=None):
        query = query or {}
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def update_one(self, query, update):
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                if "$set" in update:
                    doc.update(update["$set"])
                return
    
    def delete_one(self, query):
        for i, doc in enumerate(self._docs):
            if all(doc.get(k) == v for k, v in query.items()):
                self._docs.pop(i)
                return

    def count_documents(self, query=None):
        query = query or {}
        return sum(1 for d in self._docs
                   if all(d.get(k) == v for k, v in query.items()))

    def aggregate(self, pipeline):
        # Minimal stub — just return empty for in-memory mode
        return []


class _InMemoryDB:
    def __init__(self, patients, scans, feedback):
        self.patients = patients
        self.scans = scans
        self.feedback = feedback


# ── Try MongoDB Atlas, fall back to in-memory ────────────────────────────────
try:
    MongoClient = importlib.import_module("pymongo").MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Force connection check
    client.server_info()
    db                   = client["gastric_sentinel"]
    patients_collection  = db["patients"]
    scans_collection     = db["scans"]
    feedback_collection  = db["feedback"]
    print("[DB] Connected to MongoDB Atlas ✓")
except Exception as exc:
    print(f"[DB] MongoDB unavailable ({exc}), using in-memory fallback")
    patients_collection  = _InMemoryCollection()
    scans_collection     = _InMemoryCollection()
    feedback_collection  = _InMemoryCollection()
    db = _InMemoryDB(patients_collection, scans_collection, feedback_collection)


# ── Helper functions ─────────────────────────────────────────────────────────

def add_patient(name: str, age: int, gender: str,
                condition: str = "", risk: str = "low",
                phone: str = "", notes: str = "") -> str:
    """Insert a new patient and return the inserted _id as a string."""
    doc = {
        "name":           name.strip(),
        "age":            int(age),
        "gender":         gender,
        "condition":      condition,
        "last_diagnosis": condition,
        "risk":           risk,
        "phone":          phone,
        "notes":          notes,
        "created_at":     datetime.utcnow().isoformat(),
        "updated_at":     datetime.utcnow().isoformat(),
    }
    result = patients_collection.insert_one(doc)
    return str(result.inserted_id)


def save_scan(patient_id: str, report: dict, image_path: str = None):
    """Persist an AI scan result linked to a patient."""
    scan = {
        "patient_id":   patient_id,
        "diagnosis":    report.get("diagnosis"),
        "tier":         report.get("tier", "NEGATIVE"),
        "confidence":   report.get("confidence"),
        "risk_score":   report.get("risk_score"),
        "predicted_class": report.get("predicted_class"),
        "probabilities": report.get("probabilities"),
        "image_path":   image_path,
        "gradcam_url":  report.get("gradcam_url"),
        "created_at":   report.get("created_at", datetime.utcnow().isoformat()),
    }
    scans_collection.insert_one(scan)


def save_feedback(prediction: str, correction: str):
    """Store doctor feedback on a prediction."""
    feedback_collection.insert_one({
        "prediction":  prediction,
        "correction":  correction,
        "created_at":  datetime.utcnow().isoformat(),
    })


def get_patient_scans(patient_id: str) -> list:
    """Return all scan records for a patient (no _id field)."""
    return list(scans_collection.find({"patient_id": patient_id}, {"_id": 0}))


def get_db():
    """Return a db handle with .patients/.scans/.feedback collections."""
    return db
