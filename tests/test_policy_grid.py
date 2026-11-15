"""
Accuracy beyond the five public tickets.

Every combination of order facts (including the 7/8 and 30/31 day boundaries) is
run through both agents and compared with the reference policy. This also proves
the guardrails never veto a decision the policy considers correct.
"""
from __future__ import annotations

import itertools

import pytest

import question1.agent as q1
import question2.agent as q2
from eval.reference import decide_order

THREADS = [
    "Please help with my order.",
    "I want a refund.",
    "It arrived cracked, please replace it.",
]


def _orders():
    for ptype, days, opened, lic, cust, dmg in itertools.product(
        ["physical", "digital", "custom", "subscription"],
        [0, 1, 7, 8, 30, 31, 90],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        base = {
            "order_id": "ord_x", "sku": "X", "title": "X", "product_type": ptype,
            "delivered_days_ago": days, "opened": opened, "license_revealed": lic,
            "customized": cust, "damage_reported": dmg,
        }
        if ptype == "subscription":
            for age in (0, 29, 30, 31, 365):
                yield {**base, "subscription_age_days": age}
        else:
            yield base


def _ticket(text: str) -> dict:
    return {"ticket_id": "T-x", "order_id": "ord_x", "messages": [{"from": "customer", "text": text}]}


@pytest.fixture
def serve(monkeypatch):
    def _serve(order: dict, text: str) -> None:
        for mod in (q1, q2):
            monkeypatch.setattr(mod, "get_ticket", lambda _tid: _ticket(text))
            monkeypatch.setattr(mod, "get_order", lambda _oid: order)
    return _serve


def test_every_order_shape_matches_reference_in_both_questions(serve):
    checked = 0
    for order in _orders():
        want = decide_order(order)
        for text in THREADS:
            serve(order, text)
            for got in (q1.SupportCopilot().handle_ticket("T-x"), q2.SupportPipeline().handle_ticket("T-x")):
                assert (got.action, got.kb_article) == (want.action, want.kb_article), order
            checked += 1
    assert checked > 2000


def test_retrieval_always_surfaces_the_governing_article(serve):
    for order in _orders():
        want = decide_order(order).kb_article
        for text in THREADS:
            serve(order, text)
            pipe = q2.SupportPipeline()
            retrieved = pipe._retrieve(pipe._triage(_ticket(text), order))
            assert want in retrieved, (order, retrieved)
