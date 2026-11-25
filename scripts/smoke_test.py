"""Run against a real, populated Qdrant collection; no synthetic corpus."""
from rag_system.engine import Engine


def main():
    engine = Engine()
    for mode in ("dense", "bm25", "hybrid"):
        result = engine.search("cardiovascular health", mode=mode)
        assert result["hits"], f"No {mode} search results; check the corpus"
        assert all(hit.get("doc_id") and hit.get("text") for hit in result["hits"])
    print("Live retrieval smoke test passed")


if __name__ == "__main__":
    main()
