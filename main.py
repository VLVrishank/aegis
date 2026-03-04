"""
Aegis Responder — FastAPI backend.
- Stateless JWT Authentication
- SQLite Ledger for Documents & Answers History
- RAG Pipeline Integration
"""
# Forced refresh to rebuild Ephemeral Vector Store with new dense text data

import os
import sqlite3
import io
import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import jwt
import bcrypt

from rag_engine import build_vector_store, process_question

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Use /data for persistent storage (Hugging Face Spaces), fallback to local for dev
_data_dir = Path("/data") if Path("/data").exists() else BASE_DIR
DB_PATH   = _data_dir / "aegis_v2.db"

SECRET_KEY = os.environ.get("JWT_SECRET", "super-secret-aegis-key-2026")
ALGORITHM = "HS256"

ALGORITHM = "HS256"

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            question_id TEXT,
            question TEXT,
            answer TEXT,
            citation TEXT,
            snippet TEXT,
            confidence REAL DEFAULT 0.0,
            llm_status TEXT DEFAULT 'Verified',
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
    """)
    try:
        conn.execute("ALTER TABLE answers ADD COLUMN llm_status TEXT DEFAULT 'Verified';")
    except sqlite3.OperationalError:
        pass
    
    # Seed default admin if missing
    row = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
    if not row:
        default_hash = bcrypt.hashpw(b"aegis2024", bcrypt.gensalt()).decode("utf-8")
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", default_hash))
    conn.commit()
    conn.close()

# ── Lifecycle ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    build_vector_store()
    yield

app = FastAPI(title="Aegis Responder", lifespan=lifespan)

# ── Auth Dependency ───────────────────────────────────────────────────────────
def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return int(user_id)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token decoding failed")

# ── Models ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class RegenRequest(BaseModel):
    document_id: int
    question_id: str
    question: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/me")
async def get_me(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload.get("username")}
    except:
        raise HTTPException(status_code=401)

@app.get("/sample.csv")
async def get_sample():
    csv_content = (
        "Question_ID,Question\n"
        "Q1,What encryption standard does Aegis use for data at rest?\n"
        "Q2,Is multi-factor authentication mandatory for admin access?\n"
        "Q3,What is the incident response time for a critical SEV-1 event?\n"
        "Q4,How long does Aegis retain corporate email records?\n"
        "Q5,Does Aegis perform annual penetration testing?\n"
        "Q6,What is Aegis policy on employees using personal jetpacks inside the office?\n"
        "Q7,Does Aegis offer complimentary time travel insurance for contractors?\n"
        "Q8,What breed of office dragon is approved for the server room?\n"
        "Q9,Can employees bring emotional support dinosaurs to client meetings?\n"
        "Q10,What is the maximum allowed vampire headcount per engineering team?"
    )
    return Response(
        content=csv_content, 
        media_type="text/csv", 
        headers={"Content-Disposition": "attachment; filename=aegis_sample.csv"}
    )

@app.post("/register")
async def register(req: LoginRequest):
    conn = get_db()
    try:
        hashed = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (req.username, hashed))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Corporate ID already exists")
    finally:
        conn.close()
    return {"status": "success"}


@app.post("/login")
async def login(req: LoginRequest, response: Response):
    conn = get_db()
    row = conn.execute("SELECT id, password_hash FROM users WHERE username=?", (req.username,)).fetchone()
    conn.close()
    
    if not row or not bcrypt.checkpw(req.password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate 7-day token
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    token = jwt.encode(
        {"sub": str(row["id"]), "username": req.username, "exp": expire},
        SECRET_KEY, 
        algorithm=ALGORITHM
    )
    
    response.set_cookie(key="access_token", value=token, httponly=True, max_age=7*24*3600, samesite="lax")
    return {"status": "success"}


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "success"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: int = Depends(get_current_user)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if "Question" not in df.columns or "Question_ID" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must have 'Question_ID' and 'Question' columns")

    conn = get_db()
    # Create Document record
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (user_id, filename) VALUES (?, ?)", (user_id, file.filename))
    doc_id = cursor.lastrowid
    
    results = []
    
    # Process AI answers
    try:
        for _, row in df.iterrows():
            qid      = str(row["Question_ID"]).strip()
            question = str(row["Question"]).strip()
            rag_out  = process_question(question)

            conn.execute(
                """INSERT INTO answers 
                   (document_id, question_id, question, answer, citation, snippet, confidence, llm_status) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, qid, question,
                 rag_out["answer"], rag_out["citation"], rag_out["snippet"], rag_out.get("confidence", 0.0), rag_out.get("llm_status", "Verified"))
            )

            results.append({
                "id": qid,
                "question": question,
                "answer": rag_out["answer"],
                "citation": rag_out["citation"],
                "snippet": rag_out["snippet"],
                "confidence": rag_out.get("confidence", 0.0),
                "llm_status": rag_out.get("llm_status", "Verified"),
            })
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"RAG Engine Error: {str(e)}")

    conn.commit()
    conn.close()
    
    return JSONResponse(content={
        "document_id": doc_id,
        "filename": file.filename,
        "answers": results
    })


@app.get("/history")
async def get_history(user_id: int = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, created_at FROM documents WHERE user_id=? ORDER BY id DESC", 
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"id": r["id"], "filename": r["filename"], "created_at": r["created_at"]} for r in rows]


@app.get("/history/{doc_id}")
async def get_document(doc_id: int, user_id: int = Depends(get_current_user)):
    conn = get_db()
    doc = conn.execute("SELECT id, filename FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
        
    rows = conn.execute("SELECT question_id as id, question, answer, citation, snippet, confidence, llm_status FROM answers WHERE document_id=?", (doc_id,)).fetchall()
    conn.close()
    
    return {
        "document_id": doc["id"],
        "filename": doc["filename"],
        "answers": [dict(r) for r in rows]
    }

@app.post("/regenerate")
async def regenerate_answer(req: RegenRequest, user_id: int = Depends(get_current_user)):
    conn = get_db()
    # Verify the document belongs to the user
    doc = conn.execute("SELECT id FROM documents WHERE id=? AND user_id=?", (req.document_id, user_id)).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
        
    rag_out = process_question(req.question.strip())
    
    conn.execute(
        """UPDATE answers 
           SET answer=?, citation=?, snippet=?, confidence=?, llm_status=? 
           WHERE document_id=? AND question_id=?""",
        (rag_out["answer"], rag_out["citation"], rag_out["snippet"], 
         rag_out.get("confidence", 0.0), rag_out.get("llm_status", "Verified"), 
         req.document_id, req.question_id)
    )
    conn.commit()
    conn.close()
    
    return {
        "id": req.question_id,
        "question": req.question,
        "answer": rag_out["answer"],
        "citation": rag_out["citation"],
        "snippet": rag_out["snippet"],
        "confidence": rag_out.get("confidence", 0.0),
        "llm_status": rag_out.get("llm_status", "Verified"),
    }

