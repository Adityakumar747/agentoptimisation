from __future__ import annotations

import pytest

from eval.reference import TICKET_EXPECTATIONS
from question2.agent import SupportPipeline
from shared.kb import expanded_help_center_blob
from shared.tokens import count_tokens

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
BUDGET_PER_TICKET = 2500


def test_help_center_exceeds_context():
    assert count_tokens(expanded_help_center_blob()) > 200_000


def test_pipeline_resolves_public_tickets():
    pipe = SupportPipeline()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = pipe.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_holdout_subscription():
    got = SupportPipeline().handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


@pytest.mark.parametrize("tid", PUBLIC)
def test_points_per_ticket_within_budget(tid):
    pipe = SupportPipeline()
    pipe.handle_ticket(tid)
    assert pipe.session.total_points() <= BUDGET_PER_TICKET


def test_only_compose_calls_the_model():
    pipe = SupportPipeline()
    pipe.handle_ticket("T-1001")
    assert [c.label for c in pipe.session.calls] == ["compose"]


def test_prompt_never_contains_the_bulk_help_center():
    pipe = SupportPipeline()
    pipe.handle_ticket("T-1001")
    assert pipe.session.calls[0].input_tokens < 1000
