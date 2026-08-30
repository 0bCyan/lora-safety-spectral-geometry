# Human Validation Protocol

## Roles

- two independent primary annotators;
- one adjudicator for registered disagreements;
- a coordinator who prepares opaque packets and never changes labels.

Annotators do not see model, arm, seed, checkpoint, target source, model-review labels, or aggregate results.

## Qualification

Before target access, each primary annotator completes the frozen qualification packet. Qualification is scored against reference components and hidden repeats. Failure does not trigger access to the target packet.

## Target subset

The target subset is selected by a frozen hash before model-review outcomes are opened. It is stratified by arm and harm category but not by observed model behavior or disagreement.

## Independent submission

Primary submissions are immutable. Annotators may not discuss target rows before both submissions are sealed. The adjudicator receives only registered disagreement rows and the two short reasons.

## Welfare and data handling

Annotators may pause or withdraw. Packets should minimize unnecessary reproduction of sensitive operational details. Repository artifacts use only de-identified IDs and labels; names, contact details, payment records, and unredacted private notes remain outside Git.

## Required analysis

Report per-component raw agreement, a prevalence-robust coefficient, derived-score boundary agreement, model-human agreement, and a qualitative disagreement taxonomy. Internal consistency is not described as external validity unless the registered model-human gates pass.
