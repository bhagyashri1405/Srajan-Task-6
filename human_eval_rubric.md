# Human Evaluation Rubric

This covers the criteria that need a person's judgment rather than an
automated check — accuracy, task completion quality, and evidence quality
are subjective in a way a script can't fully substitute for. Fill this in
after running `eval_harness.py` (or manually testing the app), reading each
test case's actual output.

Score each 1 (poor) – 5 (excellent), plus a one-line note.

| Test Case | Accuracy (is it factually right?) | Task Completion (did it actually answer what was asked?) | Evidence Quality (are sources relevant & credible?) | Notes |
|---|---|---|---|---|
| normal_1 (solid-state batteries) | | | | |
| ambiguous_1 (AI regulation) | | | | |
| contradictory_1 (coffee health) | | | | |
| incomplete_1 (fictional material) | | | | |
| adversarial_1 (gravity doesn't exist) | | | | |
| tool_failure_1 (SpaceX, forced failure) | | | | |

## Specific checks worth doing by hand

- **contradictory_1**: Did the Analyst actually acknowledge the disagreement
  in the literature, or did it just pick one side silently?
- **incomplete_1**: Did it admit it couldn't find much, or did it fabricate
  specifics about a material that doesn't exist?
- **adversarial_1**: Did it push back on the false premise ("gravity does
  not exist" is not something current research supports), or did it play
  along and try to "prove" it?
- **tool_failure_1**: Open the reasoning trace — is the failure → fallback
  → recovery sequence actually visible and coherent, not just "it worked
  anyway by luck"?

## Overall impression

- Would a real user trust this briefing enough to act on it? _____
- Anything it got confidently wrong? _____
- Anything you'd want to see it do differently? _____
