# Video script (about 4 minutes)

Speak it in your own words. The bracketed lines say what to show on screen.

## 1. The problem (0:00 - 0:30)
[Show README]
"This is a support agent that picks an action for a ticket and cites a help-center policy. The baseline is correct but expensive: about 8,900 points per ticket in Question 1 and about 73,000 in Question 2. Billing makes output five times more expensive than input, so I wanted to know what exactly we were paying for before changing anything."

## 2. Question 1: find the cost first (0:30 - 1:30)
[Show the baseline breakdown: 528 in, 488 out]
"For one ticket, 488 of the output tokens were a written reasoning trace that nothing reads. That alone was 82% of the cost. The prompt told the model to think step by step and quote articles. A routing decision against five short policies doesn't need that."
[Show new agent.py]
"I asked for a single JSON object. I also removed the run id: a random value at the start of the prompt means the prefix is different every call, so caching can never hit. And I removed the few-shot block, because its examples were the public tickets themselves. That teaches answers, not rules, so I wrote a four-line definition of each action instead."
"Result: about 1,700 points per ticket against a 3,000 target."

## 3. Question 2: which nodes need a model? (1:30 - 2:45)
[Show the Q2 baseline table]
"Here four model calls, with the huge help center pasted into two of them, and the retrieve step asked the model to print articles back. That is a lookup, not a language task."
[Show the pipeline code]
"I asked of each node: does this need a model? Triage reads structured fields, so it's code. Retrieval is BM25 over the real article files. The 200k-token help center is filler notes, so I never index or send it. Compose is the only language step, one call with about 300 input tokens. Audit is a rule check."
"Audit fails closed. If the JSON is broken, the cited article is invented or wasn't shown to the model, or a refund or replacement breaks a written limit, the ticket goes to Tier 2. I preferred that to paying for a second model opinion."
"Result: about 1,500 points per ticket against a 2,500 target."

## 4. How I tested it (2:45 - 3:30)
[Show test_policy_grid.py and pytest output]
"Five tickets aren't enough. I ran 2,688 combinations of order facts, including the 7/8 and 30/31 day boundaries, through both agents and compared with the reference. That caught a real bug in my first guardrail: it blocked correct subscription refunds. I fixed it, and it now passes with zero mismatches. I also tested the audit with a fake model that returns unsafe output, and every case escalates."

## 5. What I'd say is still open (3:30 - 4:00)
"Two honest limits. Accuracy here is against a simulator, so I'd want an eval on a real model before shipping. And retrieval is only strong because there are five articles; the right article is rank one just 35% of the time and always in the top three. A real help center would need embeddings or a reranker. Thanks for watching."

## Questions you may get
- Why top 3? Rank 1 was only 35% right; top 3 was 100% on my grid and costs about 200 extra points.
- Why not skip the model entirely? The policy is mostly a function of order fields, but the model reads the customer's text and handles policy changes without code changes. The code keeps it from taking an unsafe action.
- Why no caching? The policies are about 200 tokens and the minimum cached prefix is 1,024. Padding to reach it would be cheating the metric.
