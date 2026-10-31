"""Extract legal documents, run AI review, and show metadata-based historical outcomes."""
import json, os
from pathlib import Path
import joblib
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image, ImageOps
import pytesseract
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL_PATH = BASE_DIR / "outcome_model.joblib"
METRICS_PATH = BASE_DIR / "outcome_model_metrics.json"
ALLOWED_TYPES = {"pdf", "docx", "image"}
MAX_TEXT_CHARS = 30000
_outcome_model = None


def extract_text_from_file(file_obj, file_type):
    if file_obj is None or not getattr(file_obj, "filename", ""):
        raise ValueError("Please choose a file to upload.")
    file_type = (file_type or "").lower()
    if file_type not in ALLOWED_TYPES:
        raise ValueError("Upload a PDF, DOCX, PNG, JPG or JPEG file.")
    try:
        if file_type == "pdf":
            reader = PdfReader(file_obj)
            text = "\n\n".join((p.extract_text() or "").strip() for p in reader.pages)
        elif file_type == "docx":
            doc = DocxDocument(file_obj)
            text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        else:
            with Image.open(file_obj) as img:
                text = pytesseract.image_to_string(ImageOps.exif_transpose(img).convert("RGB"))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is missing. Install it and add it to Windows PATH.") from exc
    except Exception as exc:
        raise ValueError(f"Could not read uploaded file: {exc}") from exc
    text = text.strip()
    if not text:
        raise ValueError("No readable text found. For scanned PDFs, use OCR or upload a clear image.")
    return text[:MAX_TEXT_CHARS]


def predict_historical_outcomes(case_type, court_name, year):
    global _outcome_model
    if _outcome_model is None:
        if not MODEL_PATH.exists():
            return {"available": False, "message": "Model not trained. Run: python train_outcome_model.py"}
        _outcome_model = joblib.load(MODEL_PATH)
    case_type, court_name = (case_type or "").strip(), (court_name or "").strip()
    try:
        year = int(year)
    except (TypeError, ValueError):
        raise ValueError("Enter a valid filing year.")
    if not case_type or not court_name:
        raise ValueError("Select a case type and enter the court name.")
    if not 1950 <= year <= 2035:
        raise ValueError("Year must be between 1950 and 2035.")
    row = pd.DataFrame([{"court_name": court_name, "case_type": case_type, "year": year}])
    probabilities = _outcome_model.predict_proba(row)[0]
    labels = _outcome_model.named_steps["classifier"].classes_
    ranked = sorted([{"outcome": str(label), "historical_probability": float(prob)}
                     for label, prob in zip(labels, probabilities)],
                    key=lambda x: x["historical_probability"], reverse=True)
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"available": True, "scope": "Historical disposal pattern using court, case type and year",
            "top_outcome": ranked[0]["outcome"], "outcomes": ranked[:5],
            "test_macro_f1": metrics.get("test_macro_f1"),
            "test_balanced_accuracy": metrics.get("test_balanced_accuracy"),
            "training_rows": metrics.get("train_rows"),
            "disclaimer": "Metadata-only historical estimates. This model does not assess case facts, evidence, arguments, law or merits, and does not predict what a judge will decide."}


def _ai_case_review(case_details):
    if client is None:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY to your .env file.")
    prompt = f"""Review this Indian legal case document for educational purposes only.
Do not predict guilt, innocence or the outcome of this case. Do not invent facts,
case citations or statutory sections. Distinguish stated facts from issues needing
verification. Explain that IPC/BNS applicability depends on incident date and
transitional law. Use headings: Case summary; Key facts; Legal issues; Potentially
relevant provisions (state uncertainty); Missing evidence/information; Questions
for an Indian advocate.

DOCUMENT:\n{case_details[:MAX_TEXT_CHARS]}"""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt,
        config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 1600})
    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty case review.")
    return answer


def predict_verdict(case_details, case_type=None, court_name=None, year=None):
    if not (case_details or "").strip():
        raise ValueError("No text was extracted from the uploaded file.")
    result = {"assessment": "AI-assisted case review", "analysis": _ai_case_review(case_details),
              "historical_model": {"available": False, "message": "Case metadata was not supplied."},
              "disclaimer": "Educational support only. The metadata model is not a merits-based prediction or legal advice. Consult a qualified Indian advocate."}
    if case_type and court_name and year:
        result["historical_model"] = predict_historical_outcomes(case_type, court_name, year)
    return result
