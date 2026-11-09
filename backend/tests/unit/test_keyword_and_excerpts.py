from __future__ import annotations

from app.rag.excerpts import relevant_excerpt
from app.rag.keyword_search import coverage, is_relevant, query_terms
from app.rag.retriever import fuse
from app.rag.vector_store import ScoredChunk


def test_query_terms_drop_stopwords_and_keep_identifiers():
    assert query_terms("What is the exact reorder point for NH-412?") == ["exact", "reorder", "point", "nh", "412"]
    assert query_terms("What is the?") == []


def test_coverage_requires_enough_query_terms():
    passage = "No reorder point, economic order quantity, or exact purchase recommendation is defined."
    terms = query_terms("What is the exact reorder point for NH-412?")
    assert coverage(terms, passage) == (3, 0.6)
    assert is_relevant(terms, passage)
    # A single shared word must not make an unrelated question "relevant".
    assert not is_relevant(query_terms("What is the boiling point of tungsten?"), passage)


def test_light_stemming_matches_plurals():
    assert is_relevant(query_terms("minimum order quantities"), "the minimum order quantity (MOQ)")


def chunk(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(cid, "d", "doc.pdf", 1, 0, cid, score)


def test_rrf_fusion_marks_match_types_and_keeps_true_scores():
    fused = fuse([chunk("a", 0.6), chunk("b", 0.4)], [chunk("c", 0.18), chunk("a", 0.6)], top_k=3)
    assert [(c.chunk_id, c.match) for c in fused] == [("a", "semantic+keyword"), ("c", "keyword"), ("b", "semantic")]
    assert fused[1].score == 0.18  # keyword-only hits keep their real cosine score


def test_excerpt_is_verbatim_and_focused():
    passage = (" ".join(f"Unrelated sentence number {i}." for i in range(20))
               + " Its central distribution centre (DC) is located in Pune. "
               + " ".join(f"Trailing sentence {i}." for i in range(20)))
    excerpt = relevant_excerpt(passage, "Where is the distribution centre located?", "It is in **Pune** [1].")
    assert "located in Pune" in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert excerpt.strip("… ") in passage  # never altered


def test_excerpt_of_short_passage_is_the_passage():
    assert relevant_excerpt("Short passage.", "q") == "Short passage."


def test_excerpt_inside_long_table_like_sentence():
    table = " ".join(f"NH-{i} Product{i} Supplier{i} {i} days" for i in range(100, 140)) + \
        " NH-205 Compact Mixer Bright Appliances 14 days " + " ".join(f"NH-{i} X Y {i} days" for i in range(300, 330))
    excerpt = relevant_excerpt(table, "supplier and lead time for NH-205", "Bright Appliances, 14 days [1]")
    assert "Bright Appliances" in excerpt and len(excerpt.split()) <= 50
