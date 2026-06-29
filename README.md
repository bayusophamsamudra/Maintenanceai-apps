# Maintenanceai-apps

Aplikasi desktop Question and answer dokumen AI local. Indeks PDF, Word, Excel, PowerPoint, dan file teks — dapat answer berdasarkan dokumen, lengkap dengan halaman yang bisa diklik.

Semua proses berjalan **100% offline**: model embedding dan LLM jalan di komputer sendiri, tidak ada data yang dikirim ke cloud.

---
<img width="1296" height="902" alt="image" src="https://github.com/user-attachments/assets/7e76bc38-da99-460f-8695-eae845ae5a41" />

## Fitur Utama

| Fitur | Detail |
|---|---|
| Format dokumen | PDF, DOCX, XLSX, PPTX, TXT, MD |
| Embedding | Qwen3-Embedding-0.6B (1024d) — multilingual Indonesia↔Inggris |
| LLM | qwen2.5:7b via Ollama (dapat diganti di `src/config.py`) |
| Retrieval | Hybrid vector + BM25 FTS5, fusi via Reciprocal Rank Fusion |
| Akurasi PDF | `sort=True` — memperbaiki PDF form di mana label & nilai ada di layer terpisah |
| Kecepatan | Cache matrix embedding in-memory; ~23ms warm search |
| UI | Dark theme PyQt6; streaming token; kutipan klik-untuk-buka |
| @mention | `@excel`, `@word`, atau `@namafile.pdf` untuk membatasi pencarian ke dokumen tertentu |

---

## Arsitektur

```
┌─────────────────────────────────────────────────────────────┐
│                         UI Layer                            │
│  MainWindow ─── ChatWidget ─── IngestDialog                 │
│     │               │               │                       │
│  Toolbar          Stream          Worker                    │
│  Sidebar          Token          Thread                     │
└─────────┬───────────┬───────────────┬───────────────────────┘
          │           │               │
          ▼           ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐
│   Retrieval │ │     LLM     │ │       Ingestion           │
│             │ │             │ │                          │
│  searcher   │ │ ollama_     │ │  doc_loader              │
│  ├ vector   │ │ client      │ │  ├ pdf_loader (sort=True)│
│  ├ FTS5 BM25│ │             │ │  ├ _load_docx            │
│  └ RRF fuse │ │ prompt_     │ │  ├ _load_xlsx (header    │
│             │ │ builder     │ │  │  auto-detect)         │
│  embedder   │ │             │ │  ├ _load_pptx            │
│  (Qwen3,    │ │             │ │  └ _load_text            │
│   GPU fp16) │ │             │ │                          │
│             │ │             │ │  chunker (real tokenizer)│
│  reranker   │ │             │ │  indexer (SQLite+numpy)  │
│  (optional) │ │             │ │                          │
└──────┬──────┘ └──────┬──────┘ └──────────────────────────┘
       │               │
       ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage                                │
│  data/index.db  (SQLite)                                    │
│  ├── chunks (id, file, page, text, embedding BLOB)          │
│  ├── chunks_fts (FTS5 virtual — auto-sync via triggers)     │
│  └── meta (generation counter — cache invalidation key)     │
│  data/pdfs/  (copy semua dokumen yang di-index)             │
│  models/Qwen3-Embedding-0.6B/  (weights lokal)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Alur Kerja

### 1. Indexing (satu kali per dokumen)

```
Dokumen → load_document() → pages [{"page": N, "text": "..."}]
       → chunk_pages()  → chunks [{"text", "page", "file"}]
       → embed_passages() → float32 vectors (GPU, fp16, batch 8)
       → SQLite INSERT  → chunks table + FTS5 via trigger
```

**Chunker** menggunakan tokenizer asli model embedding (bukan estimasi kata ×1.3) sehingga tidak ada chunk yang terpotong saat di-embed. Batas: 800 token.

**XLSX** mendeteksi otomatis baris header (15 baris pertama) dan mengubah setiap baris data jadi record deskriptif:
```
[NamaSheet] Kolom1: nilai | Kolom2: nilai | Kolom3: nilai
```
Ini memastikan LLM tahu konteks kolom bahkan saat hanya melihat satu chunk.

### 2. Pencarian (setiap pertanyaan)

```
Pertanyaan → embed_query() → query vector (1024d)
           → cosine similarity (numpy dot product, L2-normalized)
           → top-20 kandidat vektor
           + FTS5 BM25 keyword search (token eksak)
           → Reciprocal Rank Fusion (RRF, K=60)
           → pruning (min 3 selalu; rank 4-5 hanya jika FTS juga menemukan)
           → Identifier rescue (kode tag eksak seperti "4854-HV-0071B")
           → Fokus chunk ke baris yang mengandung kode tersebut
```

**Mengapa RRF?** Tidak perlu kalibrasi skor antar-sistem: setiap hit berkontribusi `1/(K + rank)`, sehingga item yang ditemukan KEDUANYA (vektor + BM25) naik ke atas secara alami.

### 3. Generasi Jawaban

```
Chunks terpilih → build_prompt() → prompt bilingual
               → ollama generate_stream() → token streaming
               → linkify_citations() → "[Sumber: file.pdf hal. N]" → clickable
```

LLM diperintahkan menjawab hanya dari konteks, langsung tanpa basa-basi, dengan tag sumber persis seperti nama file asli.

---

## Prasyarat

- Windows 10/11, Python 3.11
- [Ollama](https://ollama.com/download) — server LLM lokal
- NVIDIA GPU dengan CUDA (opsional tapi sangat disarankan; RTX 4060 8GB sudah teruji)

---

## Setup

### 1. Install Ollama & pull model

```bash
# Install dari https://ollama.com/download
ollama pull qwen2.5:7b
ollama serve   # jika belum jalan otomatis
```

### 2. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

# PyTorch CUDA (sesuaikan versi cu1xx dengan CUDA driver Anda):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> **Catatan:** PyQt6 di-pin ke 6.7.1. Versi 6.11.x gagal load DLL pada Python 3.11 Anaconda.

### 3. Download model embedding

Model disimpan lokal di `models/Qwen3-Embedding-0.6B/` agar tidak bergantung koneksi internet saat runtime.

```bash
# Buat folder
mkdir models\Qwen3-Embedding-0.6B

# Download file model (resume-capable jika koneksi putus):
curl -C - -L "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/model.safetensors" -o "models/Qwen3-Embedding-0.6B/model.safetensors"
curl -C - -L "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/config.json"         -o "models/Qwen3-Embedding-0.6B/config.json"
curl -C - -L "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/tokenizer.json"      -o "models/Qwen3-Embedding-0.6B/tokenizer.json"
curl -C - -L "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/modules.json"        -o "models/Qwen3-Embedding-0.6B/modules.json"
# ... file lain sesuai kebutuhan (tokenizer_config.json, special_tokens_map.json, dll.)
```

Alternatif jika HF Hub stabil di jaringan Anda:
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/Qwen3-Embedding-0.6B')"
```

### 4. Jalankan aplikasi

```bash
python -m src.main
```

---

## Penggunaan

### Menambah Dokumen

- Klik **"➕ Tambah Dokumen"** → pilih satu atau lebih file
- Klik **"📁 Tambah Folder"** → indexing seluruh folder secara rekursif
- Format didukung: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`

### Tanya Jawab

Ketik pertanyaan di kotak bawah, tekan Enter atau klik **Tanya ➤**.

Gunakan **@mention** untuk membatasi pencarian:
```
@excel berapa stok bearing 6205?
@MS_INDEX_20250307.xlsx tag 4853-ZY-7611 apa manufacturer-nya?
@word @pdf jadwal maintenance bulan Juli?
```

Alias yang tersedia: `@excel`, `@pdf`, `@word`, `@ppt`, `@text`

### Kutipan Klik

Jawaban mengandung tag `📄 namafile.pdf · hal. N` — klik untuk membuka dokumen langsung ke halaman tersebut (PDF: di viewer default; format lain: dibuka dengan aplikasi terkait).

### Hapus dari Index

Panel kiri "Dokumen ter-index" → pilih dokumen → klik **"Hapus dari index"**. File asli tetap ada di `data/pdfs/`.

---

## Konfigurasi

Edit [`src/config.py`](src/config.py) untuk menyesuaikan:

| Parameter | Default | Keterangan |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model LLM Ollama. Ganti tanpa re-index. |
| `EMBEDDING_MODEL` | Qwen3-Embedding-0.6B | Model embedding. **Re-index wajib jika diganti.** |
| `CHUNK_MAX_TOKENS` | `800` | Ukuran chunk maksimum (token). |
| `TOP_K` | `5` | Jumlah chunk konteks yang dikirim ke LLM. |
| `TOP_K_LOOKUP` | `3` | TOP_K untuk pertanyaan lookup kode/tag spesifik. |
| `RERANK_ENABLED` | `False` | Cross-encoder reranker (lebih presisi, lebih lambat). |
| `OLLAMA_KEEP_ALIVE` | `30m` | Durasi model LLM tetap di VRAM setelah pertanyaan. |

### Ganti Model LLM

```python
# src/config.py
OLLAMA_MODEL = "gemma2:9b"    # lebih cerdas, butuh ~6GB VRAM
# atau
OLLAMA_MODEL = "qwen2.5:3b"   # lebih cepat, akurasi sedikit turun
```

Tidak perlu re-index saat ganti LLM.

### Ganti Model Embedding

```python
# src/config.py
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # ringan, 384d
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"    # menengah, 768d
```

**Wajib re-index** semua dokumen setelah ganti embedding — dimensi vektor berbeda tidak kompatibel (aplikasi akan error dengan pesan jelas jika terdeteksi).

---

## Struktur Proyek

```
QNA APP/
├── src/
│   ├── main.py                  # Entry point
│   ├── config.py                # Semua konfigurasi terpusat
│   ├── ingestion/
│   │   ├── pdf_loader.py        # PyMuPDF, sort=True
│   │   ├── doc_loader.py        # DOCX/XLSX/PPTX/TXT/MD loaders
│   │   ├── chunker.py           # Paragraph chunker, real tokenizer
│   │   └── indexer.py           # SQLite + numpy, batch embed+insert
│   ├── retrieval/
│   │   ├── embedder.py          # SentenceTransformer, GPU fp16, lazy load
│   │   ├── searcher.py          # Hybrid RRF, cache matrix, identifier rescue
│   │   ├── reranker.py          # Optional CrossEncoder (OFF default)
│   │   └── mentions.py          # @-mention parser
│   ├── llm/
│   │   ├── ollama_client.py     # Streaming, think=False, warmup
│   │   └── prompt_builder.py   # Prompt bilingual + citation format
│   └── ui/
│       ├── main_window.py       # Toolbar, sidebar, status bar
│       ├── chat_widget.py       # QTextBrowser, streaming, linkify
│       ├── ingest_dialog.py     # Modal indexing, progress, cancel
│       ├── pdf_opener.py        # Buka dokumen + halaman
│       └── theme.py             # Dark theme QSS
├── data/
│   ├── index.db                 # SQLite index (chunks + FTS5)
│   └── pdfs/                   # Salinan dokumen yang di-index
├── models/
│   └── Qwen3-Embedding-0.6B/   # Weights embedding lokal
└── requirements.txt
```

---

## Catatan Teknis

### Mengapa SQLite + numpy, bukan ChromaDB?

ChromaDB 1.5.x mengalami crash access violation pada Python 3.11 di mesin ini (Rust client korupsi DB saat `add`; chroma-hnswlib segfault saat delete dari disk). SQLite + numpy exact cosine tidak bisa segfault dan untuk ribuan chunk lebih cepat dari HNSW (tidak ada overhead index tree).

### Mengapa `think=False` untuk Ollama?

Model berkapabilitas thinking (gemma4:e2b, dll.) menghabiskan seluruh token budget `num_predict` untuk reasoning tersembunyi — response balik kosong (`done_reason: length`). `think=False` membuat model menjawab langsung.

### Mengapa PDF harus `sort=True`?

PDF form engineering menyimpan label field dan nilai di layer draw terpisah. `get_text()` default mengikuti urutan draw, memisahkan label dari nilai. `sort=True` mengurutkan ulang berdasarkan posisi Y/X di halaman, mengembalikan adjacency yang benar.

### Cache Matrix Embedding

Matrix embedding dimuat ke RAM sekali saja dan di-invalidasi hanya saat tabel chunks berubah (dideteksi via counter generasi monotonic di tabel `meta`). Query warm: ~23ms. Mencegah pembacaan ratusan BLOB float32 dari SQLite setiap pertanyaan.

---

## Lisensi

Proyek pribadi / internal. Semua model embedding dan LLM tunduk pada lisensi masing-masing:
- Qwen3-Embedding-0.6B: [Apache 2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- qwen2.5:7b via Ollama: [Qwen License](https://ollama.com/library/qwen2.5)
