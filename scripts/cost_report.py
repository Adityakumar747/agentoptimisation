"""Per-call token / points breakdown.

    python scripts/cost_report.py                 # Q1: T-1001, Q2: T-1004 and T-1003
    python scripts/cost_report.py q1 T-1002
    python scripts/cost_report.py q2 T-1001 T-1002
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from question1.agent import SupportCopilot  # noqa: E402
from question2.agent import SupportPipeline  # noqa: E402
from shared.billing import CACHED_INPUT_PRICE, INPUT_PRICE, OUTPUT_PRICE  # noqa: E402

DEFAULT = {"q1": ["T-1001"], "q2": ["T-1004", "T-1003"]}


def report(track: str, ticket_id: str) -> None:
    agent = SupportCopilot() if track == "q1" else SupportPipeline()
    result = agent.handle_ticket(ticket_id)
    print(f"\n{track.upper()} {ticket_id} -> {result.action} / {result.kb_article}")
    print(f"  {'call':<14}{'input':>7}{'cached':>8}{'output':>8}{'points':>8}")
    for c in agent.session.calls:
        cached = min(c.cached_prefix_tokens, c.input_tokens)
        print(f"  {c.label:<14}{c.input_tokens:>7}{cached:>8}{c.output_tokens:>8}{c.points():>8}")
        print(
            f"  {'':<14}= {(c.input_tokens - cached)} x {INPUT_PRICE}"
            f" + {cached} x {CACHED_INPUT_PRICE} + {c.output_tokens} x {OUTPUT_PRICE}"
        )
    print(f"  total points: {agent.session.total_points()}")


def main(argv: list[str]) -> None:
    if argv and argv[0] in DEFAULT:
        for tid in argv[1:] or DEFAULT[argv[0]]:
            report(argv[0], tid)
        return
    for track, ids in DEFAULT.items():
        for tid in ids:
            report(track, tid)


if __name__ == "__main__":
    main(sys.argv[1:])
