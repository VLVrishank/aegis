"""
RAG Engine for Aegis Security Solutions.
Uses sentence_transformers for embeddings (local, free) and ChromaDB for vector search.
Embeddings are computed outside ChromaDB and passed in as raw vectors — avoids
ChromaDB's internal ONNX model download entirely.
"""

import os
import glob
from pathlib import Path
from typing import Optional, Any

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent / "data"
CHUNK_SIZE      = 400
TOP_K           = 5
DISTANCE_CUTOFF = 0.88   # Only the absolute best match must clear this for Occam filter
GROQ_MODEL      = "openai/gpt-oss-120b"

# ── Singletons ────────────────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[Any] = None
_collection:    Optional[Any] = None


def _model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, convert_to_numpy=True).tolist()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 150) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def build_vector_store() -> None:
    """
    Called once at startup. Reads all .txt files, embeds chunks with
    sentence_transformers, and stores raw embeddings in ChromaDB.
    ChromaDB is used in 'bring your own embedding' mode — no ONNX needed.
    """
    global _chroma_client, _collection

    # EphemeralClient = in-memory, no persistence, no ONNX model downloads
    _chroma_client = chromadb.EphemeralClient()
    # Do NOT pass embedding_function — we handle embeddings ourselves
    _collection = _chroma_client.create_collection(
        name="aegis_docs",
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for filepath in glob.glob(str(DATA_DIR / "*.txt")):
        source = Path(filepath).name
        raw    = Path(filepath).read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_text(raw)):
            ids.append(f"{source}_{idx}")
            texts.append(chunk)
            metadatas.append({"source": source})

    if texts:
        embeddings = _embed(texts)
        _collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )
    print(f"[RAG] Loaded {len(texts)} chunk(s) from {DATA_DIR}")


def process_question(question: str) -> dict:
    """
    Returns dict: {answer, citation, snippet}
    Occam's Razor: if best cosine distance > DISTANCE_CUTOFF, return "Not found"
    without ever calling Groq.
    """
    if _collection is None:
        return {"answer": "Vector store not initialized.", "citation": "None", "snippet": "None"}

    q_embed = _embed([question])

    results = _collection.query(
        query_embeddings=q_embed,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    top_doc      = results["documents"][0][0]
    top_meta     = results["metadatas"][0][0]
    top_distance = results["distances"][0][0]

    # Build combined context from all retrieved chunks
    all_docs   = results["documents"][0]
    all_metas  = results["metadatas"][0]
    combined_context = "\n\n---\n\n".join(
        f"[Source: {m['source']}]\n{d}" for d, m in zip(all_docs, all_metas)
    )

    # Calculate confidence percent for UI badge.
    # Cosine distance: 0.0 = identical (100%), 1.0 = highly dissimilar (0%).
    confidence = max(0.0, round((1.0 - top_distance) * 100, 1))

    # ── Occam's Razor Filter ──────────────────────────────────────────────────
    if top_distance > DISTANCE_CUTOFF:
        return {
            "answer"  : "Not found in references.",
            "citation": top_meta["source"],
            "snippet" : top_doc,
            "confidence": confidence,
            "llm_status": "Flagged",
        }

    # ── Generate via Groq ─────────────────────────────────────────────────────
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = (
        "You are an expert security compliance assistant for Aegis Security Solutions.\n"
        "You are given MULTIPLE context passages from internal policy documents.\n"
        "IMPORTANT: Read ALL passages carefully before answering. The answer may be in ANY passage, not just the first.\n\n"
        "You must classify your confidence and start your response with exactly ONE of the following tags:\n"
        "[VERIFIED] - Any passage contains the exact answer.\n"
        "[PLAUSIBLE] - The passages contain related clues allowing a reasonable but non-exact answer.\n"
        "[NULL] - NONE of the passages contain anything relevant to the question.\n\n"
        "Do not invent information. Answer concisely after the tag.\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {question}"
    )
    chat = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw_answer = chat.choices[0].message.content.strip()
    status = "Verified"

    if raw_answer.startswith("[NULL]") or raw_answer.lower() == "null" or "null" in raw_answer.lower()[:10]:
        status = "Flagged"
        raw_answer = "Not found in references."
    elif raw_answer.startswith("[PLAUSIBLE]"):
        status = "Plausible"
        raw_answer = raw_answer.replace("[PLAUSIBLE]", "").replace("]", "").strip()
    elif raw_answer.startswith("[VERIFIED]"):
        status = "Verified"
        raw_answer = raw_answer.replace("[VERIFIED]", "").replace("]", "").strip()
    else:
        # Fallback if LLM forgets the format
        if "not found" in raw_answer.lower():
            status = "Flagged"

    if status == "Flagged":
        raw_answer = "Not found in references."
        
    # Standardize empty starts
    if raw_answer.startswith("- "):
        raw_answer = raw_answer[2:]

    return {
        "answer"  : raw_answer,
        "citation": top_meta["source"],
        "snippet" : top_doc,
        "confidence": confidence,
        "llm_status": status,
    }
