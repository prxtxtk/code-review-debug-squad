"""Unit tests for ChromaDB persistence, ingestion, and similarity search."""

from __future__ import annotations

from pathlib import Path

import pytest

from vector_store import HashingEmbeddings, VectorStoreManager, format_hits_for_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"


@pytest.fixture()
def store(tmp_path: Path):
    manager = VectorStoreManager(
        persist_directory=tmp_path / "chroma",
        collection_name="test_guidelines",
        knowledge_base_dir=KNOWLEDGE_BASE,
        embeddings=HashingEmbeddings(),
        auto_ingest=True,
    )
    manager.initialize()
    yield manager
    manager.close()


def test_first_startup_auto_ingests_all_curated_sources(store: VectorStoreManager) -> None:
    status = store.status()
    assert status["ready"]
    assert status["count"] > 20
    assert set(status["sources"]) == {
        "pep8_style_guide.md",
        "python_bug_patterns.md",
        "pytest_testing_guide.md",
        "debugging_examples.md",
    }
    assert status["collection"] == "test_guidelines"


def test_similarity_search_returns_metadata_and_scores(store: VectorStoreManager) -> None:
    hits = store.search("mutable default list shared between calls", k=3)
    assert hits
    assert len(hits) <= 3
    for hit in hits:
        assert hit.content
        assert hit.source.endswith(".md")
        assert hit.section
        assert 0.0 < hit.similarity <= 1.0
        assert hit.distance >= 0.0
        assert hit.metadata.get("chunk_id")


def test_retrieved_context_contains_source_and_similarity(store: VectorStoreManager) -> None:
    hits = store.search("pytest edge cases boundary tests", k=2)
    context = format_hits_for_prompt(hits)
    assert "source:" in context
    assert "similarity:" in context
    assert "RETRIEVED 1" in context


def test_reindex_rebuilds_without_losing_source_documents(store: VectorStoreManager) -> None:
    before = store.count()
    info = store.reindex()
    assert info["reindexed"] is True
    assert info["ingested_chunks"] == before
    assert store.count() == before
    assert len(info["documents"]) == 4


def test_persistence_reopens_existing_collection(tmp_path: Path) -> None:
    directory = tmp_path / "persisted"
    first = VectorStoreManager(
        persist_directory=directory,
        collection_name="persist_test",
        knowledge_base_dir=KNOWLEDGE_BASE,
        embeddings=HashingEmbeddings(),
    )
    first.initialize()
    count = first.count()
    first.close()

    second = VectorStoreManager(
        persist_directory=directory,
        collection_name="persist_test",
        knowledge_base_dir=KNOWLEDGE_BASE,
        embeddings=HashingEmbeddings(),
    )
    second.initialize()
    try:
        assert second.count() == count
        assert second.search("docstrings", k=1)
    finally:
        second.close()
