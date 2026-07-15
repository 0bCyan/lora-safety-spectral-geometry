# Step2I-R Candidate Validation Status

## Prepared Inputs

| artifact | rows | status |
| --- | ---: | --- |
| behavior adjudication blinded table | 1200 | ready for independent judge |
| target-quality review blinded table | 80 | ready for independent judge/human review |
| semantic-template audit | 0 | pending_unavailable |

## Semantic Audit Dependency

```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "status": "pending_unavailable",
  "sentence_transformers_importable": false,
  "model_loadable_offline": false,
  "note": "Per plan, no replacement embedding model is selected after viewing Step2I-R arm results.",
  "import_error": "ModuleNotFoundError(\"No module named 'sentence_transformers'\")"
}
```

## Notes

Behavior adjudication inputs hide arm, schedule, checkpoint, current heuristic labels, and prior report decisions. Separate key files are stored for later analysis after adjudication is complete.

Target-quality review inputs hide model outcomes and contain only prompt plus target response. The current local preflight decisions remain non-independent and are not treated as approval evidence.
