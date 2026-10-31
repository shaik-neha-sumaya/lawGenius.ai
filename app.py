"""Flask application entry point for LawGenius AI."""
import hashlib
import json
import logging
import os
import secrets
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from views.chatbotLegalv2 import (create_new_chat, get_chat_list, load_chat,
                                  process_input)
from views.judgmentPred import extract_text_from_file, predict_verdict
from views.docGen import generate_legal_document

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lawgenius")


def load_users():
    try:
        with USERS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        logger.exception("Unable to read the user store")
        return {}


def save_users(users):
    temporary = USERS_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)
    temporary.replace(USERS_FILE)


def _safe_next(value):
    if not value:
        return url_for("index")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or not value.startswith("/") or value.startswith("//"):
        return url_for("index")
    return value


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)
    return wrapper


def _user_folder():
    user_key = hashlib.sha256(session["user"].encode("utf-8")).hexdigest()[:24]
    folder = BASE_DIR / "static" / "generated_docs" / user_key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@app.errorhandler(413)
def too_large(_error):
    if request.path == "/predict":
        return jsonify({"error": "The uploaded file is too large. Maximum size is 12 MB."}), 413
    return "File too large (maximum 12 MB).", 413


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or "@" not in email:
            flash("Please enter a valid email address.", "warning")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "warning")
            return redirect(url_for("register"))
        users = load_users()
        if email in users:
            flash("Email already registered. Please log in.", "warning")
            return redirect(url_for("login"))
        users[email] = {"password": generate_password_hash(password)}
        save_users(users)
        session.clear()
        session["user"] = email
        return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = load_users().get(email)
        if not user or not check_password_hash(user.get("password", ""), password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))
        session.clear()
        session["user"] = email
        return redirect(_safe_next(request.args.get("next")))
    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    user_id = session["user"]
    chat_list = get_chat_list(user_id)
    active_chat = request.args.get("chat")
    active_chat_data = None
    if active_chat:
        try:
            active_chat_data = load_chat(active_chat, user_id)
        except KeyError:
            active_chat = None
    return render_template("index.html", chat_list=chat_list,
                           active_chat=active_chat,
                           active_chat_data=active_chat_data)


@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    chat_id = create_new_chat(session["user"])
    return jsonify({"chat_name": chat_id})


@app.route("/load_chat", methods=["POST"])
@login_required
def load_chat_route():
    data = request.get_json(silent=True) or {}
    chat_name = (data.get("chat_name") or "").strip()
    if not chat_name:
        return jsonify({"error": "Chat ID is required."}), 400
    try:
        chat_data = load_chat(chat_name, session["user"])
        return jsonify({"chat_data": chat_data})
    except KeyError:
        return jsonify({"error": "Chat not found."}), 404


@app.route("/predict_page")
@app.route("/predict", methods=["GET"])
@login_required
def predict_page():
    return render_template("predict.html", chat_list=get_chat_list(session["user"]))


@app.route("/generate_page")
@app.route("/generate", methods=["GET"])
@login_required
def generate_page():
    return render_template("generate.html", chat_list=get_chat_list(session["user"]))


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    chat_name = (data.get("chat_name") or "").strip()
    user_input = (data.get("user_input") or "").strip()
    if not chat_name or not user_input:
        return jsonify({"error": "A chat and a message are required."}), 400
    try:
        answer, source = process_input(chat_name, user_input, session["user"], return_source=True)
        return jsonify({"response": answer, "source": source})
    except KeyError:
        return jsonify({"error": "Chat not found. Start a new chat."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Chat request failed")
        return jsonify({"error": str(exc) if app.debug else "The assistant is temporarily unavailable. Check the server configuration and try again."}), 500


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    uploaded = request.files.get("file")
    file_type = request.form.get("file_type", "").lower()
    try:
        text = extract_text_from_file(uploaded, file_type)
        result = predict_verdict(
            text,
            case_type=request.form.get("case_type", ""),
            court_name=request.form.get("court_name", ""),
            year=request.form.get("year", ""),
        )
        return jsonify({"text": text, "result": result})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Case assessment failed")
        return jsonify({"error": "Unable to assess this document. Check the server logs and try again."}), 500


@app.route("/generate_document", methods=["POST"])
@login_required
def generate_document():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("doc_prompt") or "").strip()
    try:
        _path, file_name = generate_legal_document(prompt, save_dir=str(_user_folder()))
        return jsonify({"download_url": url_for("download_file", filename=file_name), "file_name": file_name})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Document generation failed")
        return jsonify({"error": "Unable to generate the document. Check the server logs and try again."}), 500


@app.route("/download/<path:filename>")
@login_required
def download_file(filename):
    # send_from_directory prevents path traversal; each user has a separate directory.
    return send_from_directory(_user_folder(), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "true").lower() == "true")
