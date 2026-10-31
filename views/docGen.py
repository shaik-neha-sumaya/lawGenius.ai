"""Generate editable DOCX legal-document drafts without inventing party details."""
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches, Pt
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None


def generate_legal_document(prompt, save_dir="static/generated_docs"):
    request_text = (prompt or "").strip()
    if not request_text:
        raise ValueError("Please describe the legal document you want to generate.")
    if len(request_text) > 8000:
        raise ValueError("Please keep the document request under 8,000 characters.")
    if client is None:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY to the project .env file.")
    instruction = f"""Draft a clear, editable legal document for use in India based on the user's request.
Do not invent names, addresses, dates, case numbers, amounts, court names, statutory citations, or facts.
For missing information insert clearly marked [TO BE PROVIDED] placeholders. Do not claim the draft is legally valid or ready for filing. Add a short note at the end that a qualified Indian advocate should review it and verify current law. Use a professional document structure and concise language. Do not include markdown code fences.
USER REQUEST:\n{request_text}"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=instruction,
        config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 3500},
    )
    content = (getattr(response, "text", None) or "").strip()
    if not content:
        raise RuntimeError("The AI service returned an empty document. Please try again.")
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", request_text[:35]).strip("_").lower() or "legal_document"
    file_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}.docx"
    file_path = output_dir / file_name
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    lines = content.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if index == 0:
            doc.add_heading(stripped.lstrip("# "), level=0)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif re.match(r"^\s*(?:[-*]|\d+[.)])\s+", stripped):
            doc.add_paragraph(re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", stripped), style="List Bullet")
        else:
            doc.add_paragraph(stripped.replace("**", ""))
    doc.save(file_path)
    return str(file_path), file_name
