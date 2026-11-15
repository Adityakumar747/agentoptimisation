from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

ACTIONS = ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2")

_DEFAULT_REPLY = {
    "REFUND": "We've approved your refund.",
    "REPLACE": "We're sending a replacement.",
    "SEND_ARTICLE": "Here is the policy that applies to your request.",
    "ESCALATE_TIER2": "A specialist will review your case and follow up.",
}


_IDENTIFIER_FIELDS = ("order_id", "sku", "title")


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    action = str(data["action"]).upper()
    return TicketResolution(
        action=action,
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply") or _DEFAULT_REPLY.get(action, "")),
        raw_completion=text,
    )




SYSTEM_HEADER = (
    "You are ShopAssist, a Tier-1 e-commerce support router. "
    "Choose one action for the ticket using only the policies and order facts provided.\n"
    "Actions:\n"
    "REFUND - a policy grants a refund\n"
    "REPLACE - a policy grants a replacement shipment\n"
    "SEND_ARTICLE - a policy rules the request out; send the customer that article\n"
    "ESCALATE_TIER2 - no policy clearly grants or denies it, or it needs manual review\n"
    "If policies conflict, the more specific exclusion wins. "
    "If nothing grants the request, escalate and cite the closest policy."
)


def policy_line(article_id: str) -> str:
    """`[id] body` -- the article without its duplicated heading or markdown."""
    body = load_article(article_id).split("\n", 1)[-1]
    return f"[{article_id}] " + " ".join(body.replace("**", "").split())


def order_facts(order: dict) -> str:
    """Order snapshot minus identifiers, as compact JSON (still valid ORDER_SNAPSHOT)."""
    facts = {k: v for k, v in order.items() if k not in _IDENTIFIER_FIELDS}
    return json.dumps(facts, separators=(",", ":"))


def render_case(thread: str, order: dict) -> str:
    return f"THREAD:\n{thread}\nORDER_SNAPSHOT:\n{order_facts(order)}\nEND_SNAPSHOT"




def guardrail_violations(order: dict, action: str, article: str, known: set[str]) -> list[str]:
    """Reasons a model decision must not be auto-executed.

    These are *negative* rules lifted straight from the help-center wording
    (7-day damage window, 30-day return window, no refund after license reveal,
    no automated replacement of custom goods). They never pick an action; they
    only block money or shipments the written policy does not allow.
    """
    problems = []
    if action not in ACTIONS:
        problems.append(f"unknown action {action!r}")
    if article not in known:
        problems.append(f"unknown article {article!r}")

    if action == "REPLACE":
        if order.get("customized"):
            problems.append("custom items are not auto-replaced")
        if not order.get("damage_reported"):
            problems.append("replacement without reported damage")
        if int(order.get("delivered_days_ago", 99)) > 7:
            problems.append("outside the 7-day damage window")

    if action == "REFUND":
        ptype = order.get("product_type")
        if ptype == "subscription":
            if int(order.get("subscription_age_days", 99)) > 30:
                problems.append("subscription older than 30 days")
        elif order.get("license_revealed"):
            problems.append("license already revealed")
        elif ptype == "physical":
            if order.get("opened"):
                problems.append("opened goods are not auto-refunded")
            if int(order.get("delivered_days_ago", 99)) > 30:
                problems.append("outside the 30-day return window")
        else:
            problems.append(f"no policy refunds {ptype!r} orders automatically")
    return problems


def apply_guardrails(
    res: TicketResolution, order: dict, known: set[str]
) -> tuple[TicketResolution, list[str]]:
    """Fail closed: a decision that breaks a guardrail goes to a human."""
    problems = guardrail_violations(order, res.action, res.kb_article, known)
    if not problems:
        return res, problems
    article = res.kb_article if res.kb_article in known else ""
    return (
        TicketResolution(
            "ESCALATE_TIER2", article, _DEFAULT_REPLY["ESCALATE_TIER2"], res.raw_completion
        ),
        problems,
    )




class SupportCopilot:
    """
    ShopAssist Tier-1 copilot: one LLM call per ticket.

    The prompt is deliberately small and fully deterministic: no run id (it would
    change the prefix on every call), no chain-of-thought request (output tokens
    cost 5x input and a routing decision does not need a narrative), no few-shot
    examples (the policies already define the routing), and no fields the policy
    never looks at.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def _system_prompt(self) -> str:
        policies = "\n".join(policy_line(a) for a in article_ids())
        return (
            f"{SYSTEM_HEADER}\n\nPolicies:\n{policies}\n\n"
            'Reply with one JSON object only: {"action":"<action>","kb_article":"<policy id>",'
            '"customer_reply":"<one short sentence to the customer>"}'
        )

    def _messages(self, ticket: dict, order: dict) -> list[ChatMessage]:
        return [
            ChatMessage("system", self._system_prompt()),
            ChatMessage("user", render_case(format_thread(ticket), order)),
        ]

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        return self._messages(ticket, get_order(ticket["order_id"]))

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        completion = self.llm.chat("handle_ticket", self._messages(ticket, order))
        resolution, _ = apply_guardrails(parse_resolution(completion), order, set(article_ids()))
        return resolution
