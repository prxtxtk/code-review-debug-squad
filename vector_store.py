"""ChromaDB knowledge base with local, offline-capable embeddings.

Documents in ``knowledge_base/*.md`` are split by Markdown section, chunked, and
persisted to ``./chroma_db`` on first startup.  HuggingFace
``all-MiniLM-L6-v2`` is preferred; if the model cannot be loaded (for example on
a brand-new machine with no model cache), a deterministic local hashing embedder
keeps the app functional and offline.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB or local embeddings cannot be initialised."""


class HashingEmbeddings(Embeddings):
    """Deterministic offline fallback embedder.

    It is intentionally simple: token and character n-gram hashing into a
    fixed-size signed vector.  It is not a replacement for MiniLM, but it keeps
    retrieval and the entire GUI operational without downloads or API keys.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
        features: list[str] = tokens[:]
        for token in tokens:
            for n in (3, 4):
                if len(token) >= n:
                    features.extend(token[i : i + n] for i in range(len(token) - n + 1))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass(slots=True)
class RetrievalHit:
    content: str
    source: str
    title: str
    section: str
    distance: float
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def trace_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "section": self.section,
            "title": self.title,
            "distance": round(self.distance, 6),
            "similarity": round(self.similarity, 6),
            "chunk": self.content,
            "metadata": self.metadata,
        }


def format_hits_for_prompt(hits: Iterable[RetrievalHit]) -> str:
    """Create the exact context block injected into agent prompts."""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[RETRIEVED {index} | source: {hit.source} | section: {hit.section} | "
            f"similarity: {hit.similarity:.3f}]\n{hit.content}"
        )
    return "\n\n".join(blocks) if blocks else "(No knowledge-base context was retrieved for this request.)"


class VectorStoreManager:
    """Owns ChromaDB persistence, ingestion, and similarity search."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "coding_guidelines",
        knowledge_base_dir: str | Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings: Embeddings | None = None,
        auto_ingest: bool = True,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.knowledge_base_dir = Path(knowledge_base_dir or "knowledge_base")
        self.embedding_model = embedding_model
        self._embeddings = embeddings
        self.auto_ingest = auto_ingest
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store: Any = None
        self._client: Any = None
        self._lock = threading.RLock()
        self._embedding_backend = "custom" if embeddings is not None else "unknown"
        self._last_error = ""
        self._ingested_this_session = 0
        self._ready = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is not None:
            return self._embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._embedding_backend = f"HuggingFace: {self.embedding_model}"
        except Exception as exc:  # offline or model package unavailable
            self._last_error = (
                "HuggingFace embeddings unavailable; using deterministic local "
                f"hashing embeddings instead. ({type(exc).__name__}: {exc})"
            )
            self._embeddings = HashingEmbeddings()
            self._embedding_backend = "Deterministic local hashing (offline fallback)"
        return self._embeddings

    def _new_store(self):
        try:
            import chromadb
            from chromadb.config import Settings
            from langchain_chroma import Chroma
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError(
                "ChromaDB integration is missing. Run: pip install -r requirements.txt"
            ) from exc

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                create_collection_if_not_exists=True,
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise VectorStoreError(
                f"Could not open ChromaDB at {self.persist_directory}: {exc}"
            ) from exc

    def initialize(self) -> "VectorStoreManager":
        with self._lock:
            if self._ready:
                return self
            self._new_store()
            if self.auto_ingest and self.count() == 0:
                self.ingest(force=False)
            self._ready = True
            return self

    @property
    def store(self) -> Any:
        self.initialize()
        return self._store

    def count(self) -> int:
        if self._store is None:
            return 0
        try:
            return int(self._store._collection.count())
        except Exception:
            try:
                return len(self._store.get(include=[])["ids"])
            except Exception:
                return 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _load_markdown_documents(self) -> list[Document]:
        if not self.knowledge_base_dir.exists():
            raise VectorStoreError(f"Knowledge base folder not found: {self.knowledge_base_dir}")
        paths = sorted(self.knowledge_base_dir.glob("*.md"))
        if not paths:
            raise VectorStoreError(f"No Markdown files found in {self.knowledge_base_dir}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n### ", "\n## ", "\n\n", "\n", " "],
            keep_separator=True,
        )
        documents: list[Document] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            preamble, sections = _split_markdown_sections(text)
            if len(preamble.strip()) >= 120:
                documents.extend(
                    self._chunk_text(
                        preamble.strip(),
                        source=path.name,
                        title=_document_title(text) or path.stem.replace("_", " ").title(),
                        section="Overview",
                        splitter=splitter,
                    )
                )
            for title, section_text in sections:
                if not section_text.strip():
                    continue
                documents.extend(
                    self._chunk_text(
                        section_text.strip(),
                        source=path.name,
                        title=_document_title(text) or path.stem.replace("_", " ").title(),
                        section=title,
                        splitter=splitter,
                    )
                )
        return documents

    def _chunk_text(
        self,
        text: str,
        *,
        source: str,
        title: str,
        section: str,
        splitter: RecursiveCharacterTextSplitter,
    ) -> list[Document]:
        chunks = splitter.split_text(text)
        documents: list[Document] = []
        for index, chunk in enumerate(chunks):
            digest = hashlib.sha1(
                f"{source}|{section}|{index}|{chunk[:120]}".encode("utf-8")
            ).hexdigest()[:16]
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "title": title,
                        "section": section,
                        "chunk_index": index,
                        "chunk_id": f"{source}:{section}:{index}:{digest}",
                        "document": source,
                    },
                )
            )
        return documents

    def ingest(self, force: bool = False) -> dict[str, Any]:
        """Ingest the curated knowledge base; rebuild the collection if forced."""
        with self._lock:
            if self._store is None:
                self._new_store()
            if force:
                try:
                    self._client.delete_collection(self.collection_name)
                except Exception:
                    pass
                from langchain_chroma import Chroma

                self._store = Chroma(
                    client=self._client,
                    collection_name=self.collection_name,
                    embedding_function=self._get_embeddings(),
                    create_collection_if_not_exists=True,
                )
            documents = self._load_markdown_documents()
            if not documents:
                raise VectorStoreError("Knowledge-base ingestion produced zero chunks.")
            ids = [str(document.metadata["chunk_id"]) for document in documents]
            self._store.add_documents(documents=documents, ids=ids)
            self._ingested_this_session += len(documents)
            self._ready = True
            return {
                "ingested_chunks": len(documents),
                "documents": sorted({d.metadata["source"] for d in documents}),
                "count": self.count(),
                "embedding_backend": self._embedding_backend,
                "collection": self.collection_name,
                "persist_directory": str(self.persist_directory),
            }

    def reindex(self) -> dict[str, Any]:
        with self._lock:
            result = self.ingest(force=True)
            self._ready = True
            return {**result, "reindexed": True}

    def close(self) -> None:
        """Release Chroma's persistent client (useful for tests and shutdown)."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._store = None
            self._client = None
            self._ready = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 4) -> list[RetrievalHit]:
        if not query.strip():
            return []
        with self._lock:
            if self._store is None:
                self.initialize()
            k = max(1, min(int(k), 10))
            try:
                raw = self._store.similarity_search_with_score(query, k=k)
            except Exception as exc:
                raise VectorStoreError(f"Chroma similarity search failed: {exc}") from exc
            hits: list[RetrievalHit] = []
            for document, distance in raw:
                distance = float(distance)
                metadata = dict(document.metadata or {})
                hits.append(
                    RetrievalHit(
                        content=document.page_content,
                        source=str(metadata.get("source", "unknown")),
                        title=str(metadata.get("title", "")),
                        section=str(metadata.get("section", "")),
                        distance=distance,
                        similarity=1.0 / (1.0 + max(distance, 0.0)),
                        metadata=metadata,
                    )
                )
            return hits

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        try:
            self.initialize()
            collection_count = self.count()
            error = self._last_error
        except Exception as exc:
            return {
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
                "persist_directory": str(self.persist_directory),
                "collection": self.collection_name,
                "count": 0,
                "embedding_backend": self._embedding_backend,
                "knowledge_base_dir": str(self.knowledge_base_dir),
                "sources": [],
            }
        sources = sorted({path.name for path in self.knowledge_base_dir.glob("*.md")})
        return {
            "ready": True,
            "error": error,
            "persist_directory": str(self.persist_directory),
            "collection": self.collection_name,
            "count": collection_count,
            "embedding_backend": self._embedding_backend,
            "knowledge_base_dir": str(self.knowledge_base_dir),
            "sources": sources,
            "ingested_this_session": self._ingested_this_session,
        }


def _document_title(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _split_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split at H2 headings while keeping the heading with its section."""
    matches = list(re.finditer(r"^##\s+(.+)$", text, re.MULTILINE))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        chunk = text[match.start() : end].strip()
        sections.append((title, chunk))
    return preamble, sections


__all__ = [
    "HashingEmbeddings",
    "RetrievalHit",
    "VectorStoreError",
    "VectorStoreManager",
    "format_hits_for_prompt",
]
