"""The model proposes, the audit disposes: bad model output must fail closed."""
from __future__ import annotations

import json

import pytest

import question1.agent as q1
import question2.agent as q2


def _completion(action: str, article: str) -> str:
    return json.dumps({"action": action, "kb_article": article, "customer_reply": "ok"})


# (ticket, what a misbehaving model says, why it must not be auto-executed)
CASES = [
    ("T-1002", "REFUND", "digital-license-no-refund"),         # license already revealed
    ("T-1003", "REPLACE", "damage-replacement-7-days"),        # custom item, no auto replacement
    ("T-1001", "REPLACE", "damage-replacement-7-days"),        # nothing was damaged
    ("T-1091", "REFUND", "subscription-cooling-off"),          # day 38 subscription
    ("T-1004", "REFUND", "made-up-article"),                   # article does not exist
    ("T-1004", "GIVE_COUPON", "damage-replacement-7-days"),    # not an action
]


@pytest.mark.parametrize("tid,action,article", CASES)
def test_pipeline_fails_closed(monkeypatch, tid, action, article):
    pipe = q2.SupportPipeline()
    monkeypatch.setattr(pipe.llm, "chat", lambda *_a, **_k: _completion(action, article))
    assert pipe.handle_ticket(tid).action == "ESCALATE_TIER2"


@pytest.mark.parametrize("tid,action,article", CASES)
def test_copilot_fails_closed(monkeypatch, tid, action, article):
    bot = q1.SupportCopilot()
    monkeypatch.setattr(bot.llm, "chat", lambda *_a, **_k: _completion(action, article))
    assert bot.handle_ticket(tid).action == "ESCALATE_TIER2"


def test_pipeline_escalates_unparseable_output(monkeypatch):
    pipe = q2.SupportPipeline()
    monkeypatch.setattr(pipe.llm, "chat", lambda *_a, **_k: "Sorry, I can't help with that.")
    got = pipe.handle_ticket("T-1001")
    assert got.action == "ESCALATE_TIER2"


def test_pipeline_rejects_article_the_model_was_never_shown(monkeypatch):
    pipe = q2.SupportPipeline()
    # T-1002 retrieves the digital/refund policies. The damage article is a real
    # article but was not shown to the model, and SEND_ARTICLE trips no other rule,
    # so only the citation check can catch this.
    monkeypatch.setattr(
        pipe.llm, "chat", lambda *_a, **_k: _completion("SEND_ARTICLE", "damage-replacement-7-days")
    )
    assert pipe.handle_ticket("T-1002").action == "ESCALATE_TIER2"
