# ShopAssist cost lab: write-up

Both tracks keep the graded fields (`action`, `kb_article`) identical to the reference and cut points per ticket by 80-98%.

| | Baseline | Now | Target |
|---|---:|---:|---:|
| Q1, per ticket (T-1001..T-1004) | ~8,900 | 1,854 / 1,722 / 1,686 / 1,647 | <= 3,000 |
| Q2, per ticket | 73,443 (T-1001) | 1,659 / 1,530 / 1,473 / 1,434 | <= 2,500 |

Tests: `pytest` -> 33 passed. Run the suite with `pytest -q`. It takes about 4 minutes because the repo's own `test_help_center_exceeds_context` builds the 210k-token filler blob with a quadratic loop in `shared/kb.py`. To skip it: `pytest -q --deselect tests/test_question2.py::test_help_center_exceeds_context`.

Reproduce the numbers below with `python scripts/cost_report.py`.

## Question 1: one call per ticket

**Where the 8,900 points went.** One baseline call for T-1001 was 528 input and 488 output tokens: 1,584 + 7,320 = 8,904 points. The output is 82% of the bill, because output costs 5x input. That output was a narrative the product never reads, requested by "think step by step, write full reasoning, quote any article".

**What I changed**
- Removed the chain-of-thought request. The model now returns one JSON object. Routing against five short policies does not need a written argument.
- Removed `Run id <uuid>` from the system prompt. It changes the prefix on every call, so prefix caching could never match.
- Removed the few-shot block. Its four examples were the four public tickets, so it taught the answers rather than the rules. I replaced it with a four-line definition of what each action means.
- Sent the model only what the policy looks at: compact JSON order facts without `order_id`, `sku`, `title`, and articles as `[id] body` without the duplicated heading.
- Added a zero-token guardrail after parsing (see below).

**Breakdown, T-1001**

| call | input | cached | output | points |
|---|---:|---:|---:|---:|
| handle_ticket | 388 | 0 | 46 | 1,854 |

388 x 3 + 0 x 0.3 + 46 x 15 = 1,854. The other tickets: T-1002 1,722, T-1003 1,686, T-1004 1,647. Holdout T-1091 resolves correctly at 1,689.

## Question 2: four-node pipeline

**Where the 73,443 points went (T-1001).**

| node | input | output | points |
|---|---:|---:|---:|
| triage | 8,455 | 488 | 32,685 |
| retrieve | 596 | 46 | 2,478 |
| compose | 1,193 | 486 | 10,869 |
| audit | 6,717 | 484 | 27,411 |

Four model calls, the big help center pasted into two of them, and a "retrieve" node that asked the model to print articles back.

**What I changed.** Only *compose* needs a language model.
- **triage** (code): reads the structured order fields and spots intent words in the customer's lines.
- **retrieve** (code): BM25 over the real article files, top 3. The 210k-token "help center" is 99.9% filler notes; it is never indexed or sent. Retrieval does not need a model.
- **compose** (one LLM call): thread, order facts and the 3 retrieved policies. About 300 input tokens.
- **audit** (code): checks the JSON parses, the action is one of the four, the cited article is real *and was one of the articles the model was shown*, and that REFUND/REPLACE do not break the written limits (7-day damage window, 30-day return window, no refund after the license is revealed, no auto-replacement of custom goods). Any failure goes to ESCALATE_TIER2. This replaces a second paid model opinion with a rule check that costs 0 tokens and fails safe.

**Breakdown**

Physical fulfillment path, T-1004 (REPLACE / damage-replacement-7-days):

| call | input | cached | output | points |
|---|---:|---:|---:|---:|
| compose | 298 | 0 | 36 | 1,434 |

298 x 3 + 36 x 15 = 1,434. Triage, retrieve and audit make no model calls: 0 points.

Escalation path, T-1003 (ESCALATE_TIER2 / custom-items-no-replace):

| call | input | cached | output | points |
|---|---:|---:|---:|---:|
| compose | 296 | 0 | 39 | 1,473 |

296 x 3 + 39 x 15 = 1,473.

## How I checked accuracy

Five public tickets are not enough evidence. `tests/test_policy_grid.py` runs every combination of order facts (4 product types x delivery age 0/1/7/8/30/31/90 x opened x license x customized x damage, plus subscription ages 0/29/30/31/365) with three different thread texts, through both agents, and compares with `eval.reference`: 2,688 cases, 0 mismatches. This caught a real bug in my first draft: the license guardrail was also firing on subscriptions and would have escalated correct refunds.

`tests/test_guardrails.py` replaces the model with a stub that returns unsafe or malformed output (refund after license reveal, replace a custom item, invented article, invented action, plain prose, an article the model never saw). Every case must come out as ESCALATE_TIER2.

## Honest limits

- **Accuracy is measured against the local simulator**, which reads `ORDER_SNAPSHOT` and applies the reference rules. The tests prove the pipeline, retrieval and guardrails; they cannot prove that a real model would reason correctly from my prompt. That needs an eval on a real model.
- **The simulator fixes output size** (about 40 tokens, ~600 points), which is the floor in these numbers.
- **Caching does not apply here.** The policies are about 200 tokens and the cache needs a 1,024-token prefix. I did not pad the prompt to reach it. The prompt is stable and ordered static-first, so it becomes cacheable if the policy text grows.
- **Retrieval quality.** On the grid the governing article is rank 1 in 35% of cases, in the top 2 in 78%, and in the top 3 in 100%. With only five real articles, top-3 is what carries accuracy. A real-size help center would need embeddings or a reranker, not just BM25.
- I replaced the two `test_baseline_cost_is_high` tests, since they asserted the old cost, with budget tests.
