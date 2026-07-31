# Known original baseline quirks

These behaviors are intentionally frozen. A failure caused by one of them is a
baseline outcome and must not be repaired during a run.

- The outer query-budget check occurs between outer iterations. One production
  GRPO session can overshoot the remaining budget.
- Feedback parsing, malformed or absent content, missing log probabilities,
  retry behavior, and fallbacks retain the frozen implementation exactly.
- Existing checkpoint, learner/suffix/Trainer/optimizer state formats and save
  timing are unchanged. The official run is cold and does not resume.
- Existing task propagation and final-evaluation behavior are unchanged.
- The repository-faithful policy identifier is `Qwen/Qwen2-1.5B`; this is used
  despite inconsistent Qwen2/Qwen2.5 wording in the paper.
- With query budget 260 and no early stopping, the current loop shape is four
  outer evaluations plus four 64-completion GRPO sessions: 260 victim queries,
  255 hosted feedback calls, 640 forward/backward microsteps, and 320 optimizer
  updates. These are expectations, not a new stopping rule or assertion.
