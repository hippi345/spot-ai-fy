"""Committed fixture embedder — query lookup failures."""

import pytest

from evals.committed_npz import MissingCommittedQueryEmbeddingError
from evals.fixture_embedder import CommittedFixtureEmbedder


def test_embed_query_raises_when_query_not_committed() -> None:
    ev = CommittedFixtureEmbedder()
    with pytest.raises(MissingCommittedQueryEmbeddingError):
        ev.embed_query("this query string is not in retrieval_queries.py")
