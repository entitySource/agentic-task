"""Retrieval: chunking, column cards, hybrid search, persistence."""

from agentic_rag.knowledge import KnowledgeBase


def test_markdown_is_split_one_glossary_entry_per_chunk(knowledge):
    doc = next(d for d in knowledge.list_documents() if d["kind"] == "document")
    chunks = knowledge.document_chunks(doc["id"])
    sections = [c.get("section", "") for c in chunks]
    assert any(s.endswith("Churn and churn rate") for s in sections)
    assert any(s.endswith("Festive season") for s in sections)
    assert len(chunks) >= 12


def test_search_finds_the_right_definition(knowledge):
    hit = knowledge.search("What does churn rate mean?", k=1)[0]
    assert "Churn and churn rate" in hit.location and "cancelled" in hit.text
    hit = knowledge.search("when is the festive season", k=1)[0]
    assert "October to December" in hit.text


def test_column_cards_only_for_the_active_dataset(knowledge, subs, sales):
    knowledge.index_dataset(subs)
    knowledge.index_dataset(sales)
    hits = knowledge.search("support tickets", dataset_fp=subs.fingerprint, k=8, kinds=("dataset",))
    assert hits and all(h.title == "subscriptions columns" for h in hits)
    assert hits[0].location == "column support_tickets"
    assert not knowledge.search("support tickets", dataset_fp=None, k=5, kinds=("dataset",))


def test_reindexing_a_dataset_replaces_its_cards(knowledge, subs):
    knowledge.index_dataset(subs)
    knowledge.index_dataset(subs)
    cards = [d for d in knowledge.list_documents() if d["kind"] == "dataset"]
    assert len(cards) == 1 and cards[0]["chunks"] == len(subs.columns) + 1


def test_upload_delete_and_persistence(tmp_path, embeddings):
    path = tmp_path / "kb.db"
    kb = KnowledgeBase(path, embeddings)
    doc = kb.add_document("notes.txt", b"Gross margin is revenue minus cost of goods sold, divided by revenue.")
    assert doc["chunks"] == 1
    reopened = KnowledgeBase(path, embeddings)
    assert reopened.search("how is gross margin calculated", k=1)[0].doc_id == doc["id"]
    assert reopened.delete_document(doc["id"])
    assert not reopened.search("gross margin", k=1)


def test_rejects_unknown_and_empty_documents(knowledge):
    import pytest

    with pytest.raises(ValueError):
        knowledge.add_document("photo.png", b"\x89PNG")
    with pytest.raises(ValueError):
        knowledge.add_document("empty.txt", b"   ")
