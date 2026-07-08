"""Shared fixtures.

Every test gets an isolated graph session — no test sees another's
nodes, with or without manual graph.reset() calls (which now clear the
session's roots and remain harmless).
"""

import pytest

from motif import graph


@pytest.fixture(autouse=True)
def _graph_session():
    with graph.session():
        yield
