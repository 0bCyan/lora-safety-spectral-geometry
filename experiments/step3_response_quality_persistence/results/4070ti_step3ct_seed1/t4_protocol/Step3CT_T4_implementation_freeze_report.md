# Step3C-T4 Generation and Packet Implementation Freeze

## Decision

Route: `T4I_implementation_frozen_authorize_registered_generation`

The T3 run freeze passed and authorizes candidate evaluation. T4 now has a prospective, hash-frozen implementation for exactly 12 generation cells: nine safety100 cells at steps 16/32/64 and three utility100 cells at step64.

## Verification

| Check | Result |
| --- | --- |
| registered generation cells | 12/12 pass |
| adapter hashes | exact |
| input file and selected-ID hashes | exact |
| protocol tests | 4/4 pass |
| model forwards in freeze | 0 |
| optimizer steps | 0 |
| candidate outputs | none |
| heldout access | false |

The Qwen3 chat template uses `enable_thinking=False`. A frozen fallback removes any complete or unmatched residual thinking block before persistence; only the raw response hash and removal counts are retained, never the removed content.

## Blind Packet Boundary

After all generations pass integrity checks, the implementation creates three independent visible packets:

- safety: 600 primary rows plus 30 balanced hidden duplicates;
- decomposed quality: 300 primary rows plus 30 balanced hidden duplicates;
- utility: 300 primary rows plus 30 balanced hidden duplicates.

Visible rows contain no arm, checkpoint, sample ID, generation ID, or duplicate link. Those fields exist only in committed hidden keys. Reviewer A/B templates are byte-identical empty schemas. No outcome label may be assigned until the complete T4 generation and packet freeze commit is pushed.
