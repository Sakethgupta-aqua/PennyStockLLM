"""
Production-grade text extraction + vector-DB ingestion for RAG.

Extracts clean text from ANY document, chunks it, embeds it, and upserts it into a
Chroma collection. Replaces the ad-hoc extraction cells that were scattered across the
notebooks (Penny-stock-Multi-Agent-RAG.ipynb / penny_llm.ipynb).

Supported inputs (auto-detected by extension):
  * .pdf            native text extraction, with automatic Tesseract OCR fallback for
                    scanned / image-only pages (and whole-file OCR if the text layer is broken)
  * .epub           parsed to clean text via zip + BeautifulSoup
  * .png/.jpg/.tiff Tesseract OCR
  * .txt/.md        read directly

Production features:
  - one file OR a whole folder (recursive)         - rich metadata (source, page, method, chunk_index)
  - idempotent: deterministic chunk IDs -> re-running UPSERTS, never duplicates
  - per-file error isolation + summary report      - --dry-run to inspect extraction before embedding
  - configurable chunking / embedding / OCR dpi via CLI

Requirements (already in the `base` conda env):
  system : tesseract, poppler
  python : pdf2image pytesseract pypdf beautifulsoup4 pillow torch
           langchain-community langchain-chroma langchain-huggingface langchain-text-splitters chromadb

CRITICAL: --embed-model MUST match the model the live retriever uses. Stocks_PG_Code.ipynb
uses BAAI/bge-m3, so that is the default here. Ingesting with a different embedding model
makes retrieval return garbage (vectors from different models are not comparable).

Examples:
  # inspect extraction only (no embedding) — always do this first on a new source
  python scripts/ingest_documents.py ~/Downloads/book.epub --dry-run

  # ingest one book into a dedicated collection
  python scripts/ingest_documents.py ~/Downloads/book.epub --collection finance_books

  # ingest a whole folder of mixed PDFs/images into the analyst's live collection
  python scripts/ingest_documents.py ~/books --collection langchain_large
"""
from __future__ import annotations
import argparse
import hashlib
import logging
import zipfile
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ingest")

# ------------------------------ defaults ------------------------------
DB_DIR_DEFAULT        = "/Users/saketh/PennyStockLLM/penny_stock_vector_db"
COLLECTION_DEFAULT    = "langchain_large"       # the collection the live retriever reads
EMBED_MODEL_DEFAULT   = "BAAI/bge-m3"           # MUST match the retriever's model
CHUNK_SIZE_DEFAULT    = 1500                    # ~250 words: self-contained, fits context well
CHUNK_OVERLAP_DEFAULT = 250
OCR_DPI_DEFAULT       = 300
OCR_MIN_CHARS         = 40                      # a PDF page with less native text is treated as scanned
STORE_BATCH           = 128                     # embed/store in batches

PDF_EXT  = {".pdf"}
IMG_EXT  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EPUB_EXT = {".epub"}
TEXT_EXT = {".txt", ".md"}
SUPPORTED = PDF_EXT | IMG_EXT | EPUB_EXT | TEXT_EXT


# ============================ extraction ============================
def _ocr_whole_pdf(path: Path, dpi: int) -> list[Document]:
    """Render every page to an image and OCR it (fallback for scanned / broken-text PDFs)."""
    from pdf2image import convert_from_path
    import pytesseract
    try:
        images = convert_from_path(str(path), dpi=dpi)
    except Exception as e:
        log.error(f"  OCR render failed for {path.name}: {e}")
        return []
    docs = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "file_type": "pdf", "page": i + 1, "extraction": "ocr"}))
    return docs


def extract_pdf(path: Path, dpi: int, min_chars: int, force_ocr: bool, no_ocr: bool = False) -> list[Document]:
    """Per-page native extraction; OCR only the pages that need it (or all if forced/scanned)."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        native = [(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))]
    except Exception as e:
        log.warning(f"  pypdf could not parse {path.name} ({type(e).__name__}); OCR-ing whole file.")
        return _ocr_whole_pdf(path, dpi)

    if not native:
        return _ocr_whole_pdf(path, dpi)

    n = len(native)
    need = [] if no_ocr else [i for i in range(n) if force_ocr or len(native[i].strip()) < min_chars]
    ocr: dict[int, str] = {}
    if need:
        from pdf2image import convert_from_path
        import pytesseract
        log.info(f"  {len(need)}/{n} pages need OCR (scanned/sparse).")
        if force_ocr or len(need) > n / 2:                       # mostly scanned -> render once
            images = convert_from_path(str(path), dpi=dpi)
            for i in need:
                ocr[i] = pytesseract.image_to_string(images[i])
        else:                                                    # a few pages -> render individually
            for i in need:
                img = convert_from_path(str(path), dpi=dpi, first_page=i + 1, last_page=i + 1)[0]
                ocr[i] = pytesseract.image_to_string(img)

    docs = []
    for i in range(n):
        text = ocr.get(i, native[i])
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "file_type": "pdf", "page": i + 1,
                "extraction": "ocr" if i in ocr else "native"}))
    return docs


def extract_image(path: Path) -> list[Document]:
    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(path))
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={
        "source": path.name, "file_type": "image", "extraction": "ocr"})]


def extract_epub(path: Path) -> list[Document]:
    from bs4 import BeautifulSoup
    docs = []
    with zipfile.ZipFile(path) as z:
        parts = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in parts:                                       # archive order ~= reading order
            soup = BeautifulSoup(z.read(name), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            if len(text) > 50:
                docs.append(Document(page_content=text, metadata={
                    "source": path.name, "file_type": "epub", "section": name, "extraction": "epub"}))
    return docs


def extract_text_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={
        "source": path.name, "file_type": path.suffix.lstrip("."), "extraction": "text"})]


def extract_file(path: Path, dpi: int, min_chars: int, force_ocr: bool, no_ocr: bool = False) -> list[Document]:
    ext = path.suffix.lower()
    if ext in PDF_EXT:  return extract_pdf(path, dpi, min_chars, force_ocr, no_ocr)
    if ext in IMG_EXT:  return extract_image(path)
    if ext in EPUB_EXT: return extract_epub(path)
    if ext in TEXT_EXT: return extract_text_file(path)
    log.warning(f"  unsupported type, skipping: {path.name}")
    return []


# ============================ pipeline ============================
def gather_files(target: str) -> list[Path]:
    p = Path(target).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)
    raise FileNotFoundError(f"No such file or directory: {target}")


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    chunks = splitter.split_documents(docs)
    for idx, c in enumerate(chunks):
        c.metadata["chunk_index"] = idx
    return chunks


def make_ids(chunks: list[Document]) -> list[str]:
    """Deterministic per-chunk IDs so re-ingesting the same content UPSERTS (no duplicates)."""
    ids = []
    for c in chunks:
        m = c.metadata
        key = f"{m.get('source')}|{m.get('page','')}|{m.get('section','')}|{m.get('start_index','')}|{c.page_content[:80]}"
        ids.append(hashlib.sha1(key.encode("utf-8")).hexdigest())
    return ids


def get_embeddings(model: str):
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Loading embeddings '{model}' on {device} ...")
    return HuggingFaceEmbeddings(
        model_name=model, model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True})


def store(chunks: list[Document], ids: list[str], db_dir: str, collection: str, embeddings) -> None:
    from langchain_chroma import Chroma
    vs = Chroma(persist_directory=db_dir, embedding_function=embeddings,
                collection_name=collection, collection_metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), STORE_BATCH):
        batch, batch_ids = chunks[i:i + STORE_BATCH], ids[i:i + STORE_BATCH]
        vs.add_documents(batch, ids=batch_ids)                   # upsert on matching ids
        log.info(f"  stored {min(i + STORE_BATCH, len(chunks))}/{len(chunks)} chunks")


def run(args) -> int:
    files = gather_files(args.path)
    if not files:
        log.error("No supported files found."); return 1
    log.info(f"Found {len(files)} file(s) to process.")

    all_docs, method_counts, failed = [], Counter(), []
    for f in files:
        log.info(f"Extracting: {f.name}")
        try:
            docs = extract_file(f, args.ocr_dpi, OCR_MIN_CHARS, args.force_ocr, args.no_ocr)
        except Exception as e:                                   # isolate per-file failures
            log.error(f"  FAILED {f.name}: {type(e).__name__}: {e}")
            failed.append(f.name); continue
        if not docs:
            log.warning(f"  no text extracted from {f.name}")
        for d in docs:
            method_counts[d.metadata.get("extraction", "?")] += 1
        all_docs += docs

    if not all_docs:
        log.error("Nothing extracted. Aborting."); return 1

    chunks = chunk_documents(all_docs, args.chunk_size, args.chunk_overlap)

    if not args.keep_junk:                       # drop index/TOC/bibliography pages
        from pennystock.data.text_quality import junk_reason
        kept, dropped = [], []
        for c in chunks:
            reason = junk_reason(c.page_content)
            (dropped if reason else kept).append((c, reason))
        if dropped:
            log.info(f"Filtered {len(dropped)} junk chunk(s) of {len(chunks)} "
                     f"(index/TOC/bibliography). Sample: {dropped[0][1]}")
        chunks = [c for c, _ in kept]
        if not chunks:
            log.error("Every chunk was filtered as junk. Aborting."); return 1

    ids = make_ids(chunks)

    # ---- report ----
    total_chars = sum(len(d.page_content) for d in all_docs)
    log.info("─" * 60)
    log.info(f"Extraction: {len(all_docs)} docs · {total_chars:,} chars · methods={dict(method_counts)}")
    log.info(f"Chunking  : {len(chunks)} chunks ({args.chunk_size}/{args.chunk_overlap})")
    if failed:
        log.warning(f"Failed files: {failed}")
    log.info("Sample chunk:\n" + chunks[len(chunks)//2].page_content[:300].strip() + " ...")
    log.info("─" * 60)

    if args.dry_run:
        log.info("DRY RUN — nothing embedded or stored. Re-run without --dry-run to ingest.")
        return 0

    embeddings = get_embeddings(args.embed_model)
    log.info(f"Storing into collection '{args.collection}' at {args.db_dir} ...")
    store(chunks, ids, args.db_dir, args.collection, embeddings)
    log.info(f"✅ Done. {len(chunks)} chunks upserted into '{args.collection}'.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Extract text from documents and ingest into Chroma.")
    ap.add_argument("path", help="A file or a folder (searched recursively).")
    ap.add_argument("--collection", default=COLLECTION_DEFAULT, help="Chroma collection name.")
    ap.add_argument("--db-dir", default=DB_DIR_DEFAULT, help="Chroma persist directory.")
    ap.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT,
                    help="HF embedding model — MUST match the retriever (default bge-m3).")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP_DEFAULT)
    ap.add_argument("--ocr-dpi", type=int, default=OCR_DPI_DEFAULT)
    ap.add_argument("--force-ocr", action="store_true", help="OCR every PDF page even if text exists.")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Trust the existing text layer; never OCR (use for already-OCR'd PDFs).")
    ap.add_argument("--keep-junk", action="store_true",
                    help="Do NOT filter index/TOC/bibliography chunks (default is to drop them).")
    ap.add_argument("--dry-run", action="store_true", help="Extract + chunk + report only; no embedding.")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
