from __future__ import annotations

import pytest

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
BUDGET_PER_TICKET = 3000


def test_resolves_public_tickets():
    copilot = SupportCopilot()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = copilot.handle_ticket(tid)
        assert got.action == expected.action
        assert got.kb_article == expected.kb_article


def test_holdout_subscription():
    got = SupportCopilot().handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


@pytest.mark.parametrize("tid", PUBLIC)
def test_points_per_ticket_within_budget(tid):
    copilot = SupportCopilot()
    copilot.handle_ticket(tid)
    assert len(copilot.session.calls) == 1
    assert copilot.session.total_points() <= BUDGET_PER_TICKET


def test_system_prompt_is_identical_across_tickets():
    # A per-call id (the old `Run id <uuid>`) in the prefix makes prefix caching impossible.
    a, b = SupportCopilot(), SupportCopilot()
    assert a.build_messages("T-1001")[0].content == b.build_messages("T-1004")[0].content


def test_customer_reply_is_always_populated():
    got = SupportCopilot().handle_ticket("T-1001")
    assert got.customer_reply
