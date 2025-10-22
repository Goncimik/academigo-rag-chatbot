# =============================
# File: app.py
# =============================

import os
import json
import uuid
import numpy as np
import requests
from typing import List, Dict, Any, Tuple

import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset  

INDEX_DIR = "./data"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_MODEL = os.environ.get("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Spaces > Settings > Secrets > Add new secret

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- Utilities ----------

def _read_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i + 1, txt))
    return pages

def _read_txt(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]

def load_docs(files: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for fp in files:
        source = os.path.basename(fp)
        if fp.lower().endswith(".pdf"):
            for page, txt in _read_pdf(fp):
                if txt.strip():
                    docs.append({"id": str(uuid.uuid4()), "source": source, "page": page, "text": txt})
        elif fp.lower().endswith(".txt"):
            for page, txt in _read_txt(fp):
                if txt.strip():
                    docs.append({"id": str(uuid.uuid4()), "source": source, "page": page, "text": txt})
    return docs

# ---------- Embeddings & Index ----------

class VectorStore:
    def __init__(self):
      
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = int(self._embedder.get_sentence_embedding_dimension())

        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

            
            if self.index.d != self.dim:
                old_meta = self.meta[:]  # metinleri koru
                self.index = faiss.IndexFlatIP(self.dim)
                self.meta = []
                if old_meta:
                    vecs = self._embed([m["text"] for m in old_meta])
                    self.index.add(vecs)
                    self.meta = old_meta
                self.save()
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = []

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return vecs.astype("float32")

    def add(self, records: List[Dict[str, Any]]):
        vecs = self._embed([r["text"] for r in records])
        self.index.add(vecs)
        self.meta.extend(records)

    def reset(self):
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        qv = self._embed([query])
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            results.append({"score": float(score), **meta})
        return results


def chunk_docs(docs: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    out = []
    for d in docs:
        for ch in splitter.split_text(d["text"]):
            out.append({
                "id": str(uuid.uuid4()),
                "source": d["source"],
                "page": d.get("page", 1),
                "text": ch.strip()
            })
    return out

# ---------- Generation ----------
def call_hf_api(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.95) -> str:
    if not HF_TOKEN:
        return "[HATA] HF_TOKEN bulunamadı. Lütfen Secrets kısmına ekleyin."
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "return_full_text": False}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        if r.status_code == 404:
            return f"[HATA] Model bulunamadı (404). HF_MODEL='{HF_MODEL}'. " \
                   f"Settings ▸ Variables içinde doğru bir model adı verin (örn: HuggingFaceH4/zephyr-7b-beta)."
        return f"[HATA] HF API {r.status_code}: {r.text}"
    data = r.json()
    if isinstance(data, list) and len(data) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    return str(data)


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    preamble = "You are an academic assistant for theses and papers. Answer in Turkish with [source:page] citations.\n\n"
    ctx = "\n\n".join([f"[Source: {c['source']} | Page: {c.get('page','?')}]\n{c['text']}" for c in contexts])
    return f"{preamble}Soru: {question}\n\nBağlam:\n{ctx}\n\nYanıt (kaynakları [source:page] ile göster):"

# ---------- HF DATASET LOADER (NEW) ----------

VS = VectorStore()

def ui_load_hf_dataset(dataset_id, split_spec, text_field, limit_items, chunk_size, chunk_overlap):
   
    try:
        ds = load_dataset(dataset_id, split=split_spec)
    except Exception as e:
        return "", f"[HATA] Dataset indirilemedi: {e}"

    texts = []
    for i, row in enumerate(ds):
        if text_field not in row or row[text_field] is None:
            continue
        txt = str(row[text_field]).strip()
        if txt:
            texts.append({"id": str(uuid.uuid4()), "source": f"{dataset_id}", "page": 1, "text": txt})
        if limit_items > 0 and len(texts) >= int(limit_items):
            break

    if not texts:
        return "", "[HATA] Metin bulunamadı. text_field adını kontrol edin."

    chunks = chunk_docs(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    VS.add(chunks)
    VS.save()
    summary = {"rows": len(ds), "loaded_docs": len(texts), "chunks": len(chunks), "total_vectors": VS.index.ntotal}
    return json.dumps(summary, ensure_ascii=False), "HF dataset yüklendi ve indekslendi."

# ---------- Gradio UI ----------

def ui_reset_index():
    VS.reset()
    return gr.update(value=""), gr.update(visible=True), "Index sıfırlandı."

def ui_build_index(files, chunk_size, chunk_overlap):
    if not files:
        return "", "Lütfen dosya yükleyin.", None
    paths = [f.name for f in files]
    docs = load_docs(paths)
    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    VS.add(chunks)
    VS.save()
    return json.dumps({"docs": len(docs), "chunks": len(chunks)}, ensure_ascii=False), "İndeks oluşturuldu.", None

def ui_ask(question, top_k, temperature, max_tokens):
    if VS.index.ntotal == 0:
        return "Önce index oluşturun veya dataset yükleyin.", []
    retrieved = VS.search(question, k=top_k)
    prompt = build_prompt(question, retrieved)
    answer = call_hf_api(prompt, max_new_tokens=int(max_tokens), temperature=temperature)
    src_table = [[r["source"], r.get("page", "?"), f"{r['score']:.3f}", r["text"][:300] + ("…" if len(r["text"])>300 else "")] for r in retrieved]
    return answer, src_table

with gr.Blocks(title="Akademigo RAG Chatbot") as demo:
    gr.Markdown("# 🎓 Akademigo RAG Chatbot\nTez & makale QA için.")

    # NEW: HF Dataset sekmesi
    with gr.Tab("0) HF Dataset"):
        with gr.Row():
            dataset_id = gr.Textbox(value="ccdv/pubmed-summarization", label="Dataset ID")
            split_spec = gr.Textbox(value="train[:1%]", label="Split (küçük dilim önerilir)")
            text_field = gr.Textbox(value="article", label="Text field (ör. article/abstract)")
        with gr.Row():
            limit_items = gr.Slider(0, 2000, value=400, step=50, label="Maks. satır (0=limitsiz, CPU için 400 önerilir)")
            d_chunk_size = gr.Slider(256, 1500, value=800, step=32, label="Chunk size")
            d_chunk_overlap = gr.Slider(0, 400, value=120, step=8, label="Chunk overlap")
        btn_load_ds = gr.Button("İndir & İndeksle")
        ds_json = gr.Textbox(label="Özet", interactive=False)
        ds_status = gr.Markdown()
        btn_load_ds.click(
            ui_load_hf_dataset,
            inputs=[dataset_id, split_spec, text_field, limit_items, d_chunk_size, d_chunk_overlap],
            outputs=[ds_json, ds_status]
        )

    with gr.Tab("1) Index Kur"):
        up = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], label="PDF/TXT yükle")
        with gr.Row():
            chunk_size = gr.Slider(256, 1500, value=800, step=32, label="Chunk size")
            chunk_overlap = gr.Slider(0, 400, value=120, step=8, label="Chunk overlap")
        btn_build = gr.Button("İndeksle")
        btn_reset = gr.Button("Sıfırla", variant="stop")
        build_json = gr.Textbox(label="Özet", interactive=False)
        build_status = gr.Markdown()
        btn_build.click(ui_build_index, inputs=[up, chunk_size, chunk_overlap], outputs=[build_json, build_status, up])
        btn_reset.click(ui_reset_index, outputs=[up, up, build_status])

    with gr.Tab("2) Soru Sor"):
        q = gr.Textbox(label="Soru", placeholder="Sorunuzu yazın...")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
        temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
        max_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
        ask_btn = gr.Button("Yanıtla")
        ans = gr.Markdown()
        src = gr.Dataframe(
            headers=["Source", "Page", "Score", "Snippet"],
            datatype=["str", "str", "number", "str"]
        )
        ask_btn.click(ui_ask, inputs=[q, top_k, temperature, max_tokens], outputs=[ans, src])

if __name__ == "__main__":
    demo.launch()

    with gr.Tab("2) Soru Sor"):
        q = gr.Textbox(label="Soru", placeholder="Sorunuzu yazın...")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
        temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
        max_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
        ask_btn = gr.Button("Yanıtla")
        ans = gr.Markdown()
        src = gr.Dataframe(
            headers=["Source", "Page", "Score", "Snippet"],
            datatype=["str", "str", "number", "str"]
        )
        ask_btn.click(ui_ask, inputs=[q, top_k, temperature, max_tokens], outputs=[ans, src])

