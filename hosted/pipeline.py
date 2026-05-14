"""
GrowPak Agriculture Pipeline
STT  : Fine-tuned Whisper via Hugging Face Inference API (no local model needed)
LLM  : Groq API  (query enhancement + response generation)
RAG  : ChromaDB  (persistent vector store)
TTS  : Google Cloud TTS (Urdu WaveNet)
"""

import os
import json
import time
import base64
import warnings
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG  (override via environment variables)
# ─────────────────────────────────────────────────────────────
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH",   "./agriculture_chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME",  "agriculture_kb")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
HF_TOKEN         = os.getenv("HF_TOKEN")
HF_MODEL_ID      = os.getenv("HF_MODEL_ID", "YOUR_HF_USERNAME/whisper-urdu-growpak")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ur")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

AUDIO_OUT_DIR = "./audio_responses"
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# GLOBAL SINGLETONS  (loaded once when module is imported)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("GrowPak Pipeline — Loading models...")
print("=" * 60)

# ChromaDB
print("  Loading ChromaDB...")
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection    = _chroma_client.get_collection(name=COLLECTION_NAME)
print(f"  ✅ ChromaDB: {_collection.count()} documents")

# Embedding model
print("  Loading embedding model...")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  ✅ Embedding model: {EMBEDDING_MODEL}")

# Groq client
print("  Connecting to Groq...")
_groq_client = Groq(api_key=GROQ_API_KEY)
print(f"  ✅ Groq model: {GROQ_MODEL}")

# HF Inference API for Whisper — no local loading needed
_hf_asr_url     = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
_hf_headers     = {"Authorization": f"Bearer {HF_TOKEN}"}
print(f"  ✅ Whisper via HF Inference API: {HF_MODEL_ID}")

print("=" * 60)
print("All models ready.")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STT — Fine-tuned Whisper via HF Inference API
# ─────────────────────────────────────────────────────────────
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using your fine-tuned Whisper model hosted on
    Hugging Face Inference API. Handles cold starts with retry logic.
    Returns the transcribed text string.
    """
    ext = Path(audio_path).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise ValueError(f"Unsupported audio format: {ext}")

    params = {}
    if WHISPER_LANGUAGE:
        params = {"language": WHISPER_LANGUAGE}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Retry up to 3 times — HF cold starts return 503 for ~20s
    for attempt in range(3):
        response = requests.post(
            _hf_asr_url,
            headers=_hf_headers,
            data=audio_bytes,
            params=params,
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            # HF ASR returns {"text": "..."} 
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip()
            # Fallback if shape differs
            return str(result).strip()

        elif response.status_code == 503:
            # Model is loading (cold start) — wait and retry
            wait = 20 if attempt == 0 else 10
            print(f"[STT] HF model loading, waiting {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)

        else:
            raise RuntimeError(
                f"HF Inference API error {response.status_code}: {response.text}"
            )

    raise RuntimeError("HF Inference API failed after 3 attempts — model may still be loading.")


# ─────────────────────────────────────────────────────────────
# LLM — Groq (replaces Ollama)
# ─────────────────────────────────────────────────────────────
def _call_groq(prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
    """Call Groq API with a prompt and optional system message."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


def enhance_farmer_query(raw_question: str) -> Dict:
    """
    LLM Step 1: Translate Roman Urdu/Punjabi → structured English query for RAG.
    Returns a dict with enhanced_query, crop, topic, keywords, etc.
    """
    system_prompt = """You are a translation-first agricultural query normalizer for Pakistani farmer questions.
You understand Roman Urdu, Roman Punjabi, Urdu terms written in Latin script and Urdu/Arabic script, and English.

NON-NEGOTIABLE RULES:
- Translate faithfully. Do not change crop, input, disease, pest, timing, quantity, or intent.
- Never substitute one crop/input for another.
- If a source term is ambiguous, keep it in parentheses in enhanced_query.
- Do not invent missing details. Use Unknown or Not specified when unclear.
- Prefer literal correctness over fluent rewriting.
- Output STRICT JSON only (no markdown, no extra text)."""

    enhancement_prompt = f"""Farmer question: "{raw_question}"

Return EXACTLY one JSON object with these keys:
1. "enhanced_query": precise English translation used for retrieval
2. "detected_language": one of ["Roman Urdu", "Roman Punjabi", "Urdu", "Punjabi", "English", "Mixed"]
3. "crop": one of common crops or "Unknown"
4. "topic": one of [Sowing, Seed Rate, Nursery, Transplanting, Fertilizer, Irrigation, Weed Management, Pest Management, Disease Management, Harvesting, Yield, Variety, Soil, Climate, Land Preparation, General]
5. "stage": one of [Pre-Sowing, Nursery, Germination, Vegetative, Flowering, Fruit Formation, Grain Formation, Maturity, Harvest, Post-Harvest, Any]
6. "intent_type": one of [Recommendation, Prevention, Chemical Control, Biological Control, Cultural Control, Mechanical Control, Symptoms, Impact, Duration, Identification, Dosage, Timing, Fact, Comparison]
7. "entity": specific disease/pest/fertilizer/practice or "Not specified"
8. "keywords": list of 3-5 short English search keywords, lowercase, no duplicates
9. "season": one of [Kharif, Rabi, Spring, Summer, Winter, Not specified]
10. "reply_language": always "English"
11. "translation_confidence": number between 0 and 1
12. "ambiguity_notes": short string; "None" if clear

Return JSON only."""

    _DEFAULTS = {
        "enhanced_query": raw_question,
        "detected_language": "Unknown",
        "crop": "Unknown",
        "topic": "General",
        "stage": "Any",
        "intent_type": "Fact",
        "entity": "Not specified",
        "keywords": raw_question.split()[:5],
        "season": "Not specified",
        "reply_language": "English",
        "translation_confidence": 0.5,
        "ambiguity_notes": "None",
    }

    try:
        raw = _call_groq(enhancement_prompt, system_prompt, temperature=0.1)
        # Strip markdown fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        enhanced = json.loads(raw.strip())

        for k, v in _DEFAULTS.items():
            if k not in enhanced or enhanced[k] in (None, ""):
                enhanced[k] = v

        if not isinstance(enhanced.get("keywords"), list):
            enhanced["keywords"] = str(enhanced["keywords"]).split()[:5]

        seen, normalized = set(), []
        for kw in enhanced.get("keywords", []):
            kw_s = str(kw).strip().lower()
            if kw_s and kw_s not in seen:
                seen.add(kw_s)
                normalized.append(kw_s)
        enhanced["keywords"] = normalized[:5]

        return enhanced

    except Exception as e:
        print(f"[WARNING] Query enhancement failed ({e}), using fallback.")
        return _DEFAULTS


# ─────────────────────────────────────────────────────────────
# RAG — ChromaDB vector search
# ─────────────────────────────────────────────────────────────
def rag_search(enhanced_query: Dict, top_k: int = 5) -> List[Dict]:
    """
    Embed the enhanced English query + keywords and search ChromaDB.
    Applies crop/topic metadata filter when available; falls back to no filter.
    """
    search_text = enhanced_query.get("enhanced_query", "")
    if enhanced_query.get("keywords"):
        search_text += " " + " ".join(enhanced_query["keywords"])
    if enhanced_query.get("entity") and enhanced_query["entity"] != "Not specified":
        search_text += " " + enhanced_query["entity"]

    query_embedding = _embedding_model.encode(search_text).tolist()

    crop  = enhanced_query.get("crop", "Unknown")
    topic = enhanced_query.get("topic", "General")

    where_filter = {}
    if crop not in ("Unknown", "Not specified", "") and topic not in ("General", "Not specified", ""):
        where_filter = {"$and": [{"crop": {"$eq": crop}}, {"topic": {"$eq": topic}}]}
    elif crop not in ("Unknown", "Not specified", ""):
        where_filter = {"crop": {"$eq": crop}}

    try:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
        )
    except Exception as e:
        print(f"[WARNING] Filtered search failed ({e}), retrying without filter...")
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    formatted = []
    if results and results["metadatas"]:
        for i, meta in enumerate(results["metadatas"][0]):
            formatted.append({
                "question":    meta["question"],
                "answer":      meta["answer"],
                "crop":        meta["crop"],
                "topic":       meta["topic"],
                "stage":       meta["stage"],
                "intent_type": meta["intent_type"],
                "entity":      meta["entity"],
                "similarity":  round(1 - results["distances"][0][i], 4),
            })
    return formatted


# ─────────────────────────────────────────────────────────────
# LLM — Response generation (Groq)
# ─────────────────────────────────────────────────────────────
def generate_farmer_response(
    original_question: str,
    rag_results: List[Dict],
    enhanced_query: Dict,
) -> Dict:
    """
    Generate a natural Urdu-script response for the farmer.
    Uses RAG context if available, falls back to pure LLM.
    Returns {"raw_rag_answer": str|None, "refined_answer": str}.
    """
    if rag_results:
        raw_rag_answer = rag_results[0]["answer"]
        context_parts  = []
        for i, r in enumerate(rag_results[:3]):
            context_parts.append(
                f"Source {i+1} (Crop: {r['crop']}, Topic: {r['topic']}, "
                f"Similarity: {r['similarity']}):\n"
                f"Q: {r['question']}\nA: {r['answer']}"
            )
        context = "\n\n".join(context_parts)
        kb_instruction = "Use the knowledge base information above to answer."
    else:
        raw_rag_answer = None
        context        = "No relevant information was found in the knowledge base."
        kb_instruction = (
            "No knowledge base match was found. Answer from your own agricultural knowledge. "
            "Be honest if you are not certain."
        )

    system_prompt = """You are a helpful agricultural advisor for Pakistani farmers.
You are warm, practical, and speak like a knowledgeable local expert.
Always reply in simple, easy Urdu script that a rural farmer can understand.
For chemical names, dosages, and numbers, always write them in English as they are — do not translate them.
Example: "گندم کی زنگ کے لیے Propiconazole 1ml فی لیٹر پانی میں ملا کر spray کریں۔" """

    response_prompt = f"""A Pakistani farmer asked: "{original_question}"
Intent: {enhanced_query.get('intent_type', 'Recommendation')}

Knowledge base context:
{context}

Rules:
- Reply in simple, easy Urdu script. Use plain everyday Urdu a farmer would understand.
- Chemical names, medicine names, dosages, and numbers must stay in English exactly as they are (e.g. "DAP 50kg", "Chlorpyrifos 2.5ml").
- 1-2 sentences MAXIMUM
- Start directly with the answer, no preamble
- Only include: what to do, what chemical/dose if available
- IMPORTANT: If a piece of information describes CONDITIONS that cause or spread a disease, do NOT present it as treatment timing.
- If the KB does not contain specific treatment info, give advice if you are aware of the thing being talked about.
- No filler, no restating the question

Answer:"""

    try:
        refined_answer = _call_groq(response_prompt, system_prompt, temperature=0.3)
        return {"raw_rag_answer": raw_rag_answer, "refined_answer": refined_answer}
    except Exception as e:
        print(f"[ERROR] LLM generation failed ({e}) — returning raw RAG answer.")
        fallback = raw_rag_answer or "معافی کریں، ابھی جواب دینے میں دقت ہو رہی ہے۔"
        return {"raw_rag_answer": raw_rag_answer, "refined_answer": fallback}


# ─────────────────────────────────────────────────────────────
# TTS — Google Cloud TTS (Urdu)
# ─────────────────────────────────────────────────────────────
def text_to_speech_urdu(text: str, output_path: str = None) -> str:
    """
    Convert Urdu text to speech using Google Cloud TTS.
    Saves as .mp3 and returns the file path.
    """
    if output_path is None:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{AUDIO_OUT_DIR}/response_{timestamp}.mp3"

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": "ur-IN",
            "name": "ur-IN-Chirp3-HD-Aoede",
            "ssmlGender": "FEMALE",
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.9,
            "pitch": 0.0,
            "volumeGainDb": 0.0,
        },
    }

    response = requests.post(url, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Google TTS error {response.status_code}: {response.text}")

    audio_content = response.json().get("audioContent")
    if not audio_content:
        raise RuntimeError("No audioContent returned from Google TTS.")

    audio_bytes = base64.b64decode(audio_content)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(audio_path: str = None, text_input: str = None) -> Dict:
    """
    Run the full STT → Query Enhancement → RAG → LLM → TTS pipeline.
    Provide exactly one of audio_path or text_input.
    Returns a dict with all intermediate results and final audio path.
    """
    if bool(audio_path) == bool(text_input):
        raise ValueError("Provide exactly one of audio_path or text_input.")

    result = {}

    # 1. STT
    if audio_path:
        result["transcribed_text"] = transcribe_audio(audio_path)
        farmer_text = result["transcribed_text"]
    else:
        farmer_text = str(text_input).strip()
        result["transcribed_text"] = None

    result["farmer_text"] = farmer_text
    print(f"[STT] {farmer_text}")

    # 2. LLM Query Enhancement
    enhanced_query = enhance_farmer_query(farmer_text)
    result["enhanced_query"] = enhanced_query
    print(f"[ENHANCE] {enhanced_query.get('enhanced_query')}")

    # 3. RAG Search
    rag_results  = rag_search(enhanced_query, top_k=5)
    good_results = [r for r in rag_results if r["similarity"] >= SIMILARITY_THRESHOLD]
    result["rag_results"]  = rag_results
    result["good_results"] = good_results
    result["using_rag"]    = bool(good_results)
    print(f"[RAG] {len(good_results)}/{len(rag_results)} results above threshold")

    # 4. LLM Response
    llm_out = generate_farmer_response(farmer_text, good_results, enhanced_query)
    result["raw_rag_answer"] = llm_out["raw_rag_answer"]
    result["final_answer"]   = llm_out["refined_answer"]
    print(f"[LLM] {result['final_answer'][:80]}...")

    # 5. TTS
    try:
        tts_path = text_to_speech_urdu(result["final_answer"])
        result["audio_response"] = tts_path
        print(f"[TTS] Saved → {tts_path}")
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        result["audio_response"] = None

    return result
