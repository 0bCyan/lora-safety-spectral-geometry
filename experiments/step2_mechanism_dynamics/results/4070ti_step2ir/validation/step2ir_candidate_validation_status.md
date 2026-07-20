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
  "python_version": "3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.26200-SP0",
  "sentence_transformers_version": "5.6.0",
  "torch_version": "2.5.1+cu121",
  "error": "OSError(\"We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.\")",
  "note": "Exact model unavailable; no replacement embedding model was selected."
}
```

Semantic audit runner prepared: `scripts/run_step2ir_semantic_template_audit.py`. Run it again after the exact model snapshot is available locally.

## Notes

Behavior adjudication inputs hide arm, schedule, checkpoint, current heuristic labels, and prior report decisions. Separate key files are stored for later analysis after adjudication is complete.

Target-quality review inputs hide model outcomes and contain only prompt plus target response. The current local preflight decisions remain non-independent and are not treated as approval evidence.
