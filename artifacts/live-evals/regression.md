# Agent eval regression report

**Status:** fail

## Regressions

- case_count mismatch: ran 6 cases, baseline expects 13

## Candidate metrics

```json
{
  "case_count": 6,
  "assertion_pass_rate": {
    "MaxDuration": 1.0,
    "ToolSequenceMatch": 1.0,
    "ToolSequencePrefix": 1.0,
    "ForbiddenTools": 1.0,
    "ShortcutRouting": 1.0,
    "HitAt1": 1.0,
    "CaseMaxLatency": 1.0,
    "ReplyGrounding": 1.0,
    "ToolErrorHandled": 1.0,
    "MockScriptRan": 1.0,
    "MoodIntentJudge": 1.0,
    "ToolJsonValid": 1.0,
    "accurate_summary_pass": 0.5,
    "concise_friendly_pass": 1.0,
    "no_false_claims_pass": 0.8333333333333334
  },
  "score_means": {
    "accurate_summary_score": 0.5,
    "concise_friendly_score": 1.0,
    "no_false_claims_score": 0.8333333333333334
  }
}
```
