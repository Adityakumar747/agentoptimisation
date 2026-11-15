from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import (
    SYSTEM_HEADER,
    TicketResolution as _FullResolution,
    apply_guardrails,
    parse_resolution,
    policy_line,
    render_case,
)

TOP_K = 3


@dataclass
class TicketResolution:
    action: str
    kb_article: str


# --------------------------------------------------------------------------- #
# Retrieval index: BM25 over the real help-center articles
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its me my of on or our "
    "that the their they this to was were will with you your".split()
)


def _terms(text: str) -> list[str]:
    # Truncating to 6 chars is a cheap stemmer: damage/damaged, replace/replacement,
    # refund/refundable, custom/customized all collapse to one term.
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:6] for w in words if w not in _STOPWORDS]


class ArticleIndex:
    """Tiny BM25 index. Built once per process from the article files."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = {
            aid: Counter(_terms(aid.replace("-", " ") + " " + load_article(aid)))
            for aid in article_ids()
        }
        self.avg_len = sum(map(sum, (d.values() for d in self.docs.values()))) / len(self.docs)
        df = Counter(t for d in self.docs.values() for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        q = set(_terms(query))
        scored = []
        for aid, tf in self.docs.items():
            length = sum(tf.values())
            score = 0.0
            for t in q & tf.keys():
                norm = tf[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf[t] * tf[t] * (self.k1 + 1) / norm
            if score > 0:
                scored.append((aid, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


@lru_cache
def _index() -> ArticleIndex:
    return ArticleIndex()


# Order field -> the vocabulary a help-center article would use for it.
_FACT_TERMS = {
    ("product_type", "subscription"): "subscription cancelled prorated",
    ("product_type", "digital"): "digital license download",
    ("product_type", "custom"): "custom personalized",
    ("product_type", "physical"): "physical items refund returns",
    ("license_revealed", True): "license key revealed refundable",
    ("customized", True): "custom personalized merchandise replacement",
    ("damage_reported", True): "damaged replacement shipment",
    ("opened", True): "opened goods",
    ("opened", False): "resalable refund",
}

_INTENT_TERMS = {
    "refund": r"refund|money back|reimburse|cancel",
    "replacement": r"replace|exchange|send another",
    "damage": r"crack|broke|damag|defect|shatter",
}


@dataclass
class Triage:
    """Structured read of the ticket. Built from order data + the customer's words."""

    intents: list[str]
    query: str


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class SupportPipeline:
    """
    triage -> retrieve -> compose -> audit.

    Only *compose* needs a language model. The other three nodes are local:

    * triage   reads structured order fields and spots intent keywords in the thread
    * retrieve ranks the real help-center articles (BM25) and keeps the top few
    * compose  one small LLM call: thread + facts + retrieved policies -> decision
    * audit    schema check, citation check, and policy guardrails; fails closed
               to ESCALATE_TIER2 instead of paying for a second model opinion
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.index = _index()

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        triage = self._triage(ticket, order)
        retrieved = self._retrieve(triage)
        draft = self._compose(thread, order, retrieved)
        final = self._audit(draft, order, retrieved)
        return TicketResolution(final.action, final.kb_article)

    def _triage(self, ticket: dict, order: dict) -> Triage:
        # Agent lines are boilerplate; only the customer's words carry intent.
        customer_text = " ".join(
            m["text"] for m in ticket["messages"] if m["from"] == "customer"
        ).lower()
        intents = [name for name, pat in _INTENT_TERMS.items() if re.search(pat, customer_text)]
        parts = [words for (field, value), words in _FACT_TERMS.items() if order.get(field) == value]
        return Triage(intents, " ".join(parts) + " " + customer_text)

    def _retrieve(self, triage: Triage) -> list[str]:
        return [aid for aid, _ in self.index.search(triage.query)]

    def _compose(self, thread: str, order: dict, retrieved: list[str]) -> str:
        policies = "\n".join(policy_line(a) for a in retrieved)
        system = (
            f"{SYSTEM_HEADER}\n\n"
            'Reply with one JSON object only: {"action":"<action>","kb_article":"<policy id>"}'
        )
        user = f"POLICIES:\n{policies}\n{render_case(thread, order)}"
        return self.llm.chat("compose", [ChatMessage("system", system), ChatMessage("user", user)])

    def _audit(self, draft: str, order: dict, retrieved: list[str]) -> _FullResolution:
        try:
            res = parse_resolution(draft)
        except (ValueError, KeyError, json.JSONDecodeError):
            # Unreadable output: nothing to trust, route to a human with our best article.
            return _FullResolution("ESCALATE_TIER2", retrieved[0] if retrieved else "", "", draft)
        known = set(retrieved)  # citing a policy the model was never shown is a red flag
        final, _problems = apply_guardrails(res, order, known)
        return final

