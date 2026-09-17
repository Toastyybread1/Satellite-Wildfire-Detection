# GPU Validation Tuning Review

Source: private Kaggle run 350640755, September 17, 2026. Runtime 22m 30s.
The saved log ends with `VALIDATION TUNING COMPLETE. No test evaluation performed.`
Inputs remain on Kaggle. Downloaded reports are under
`outputs/reports/tuning/wildfire_tuning/` (ignored by Git).

## Results

- `shallow_longer`: 16 filters, BCE positive weight 1, validation Dice 0.301065,
  best epoch 9 of 17, threshold 0.12, 27,361 parameters.
- `shallow_wider`: 32 filters, BCE positive weight 1, validation Dice 0.305007,
  best epoch 7 of 15, threshold 0.09, 104,897 parameters. Selected winner.
- `shallow_wider_weighted`: 32 filters, BCE positive weight 6, validation Dice
  0.303221, best epoch 11 of 19, threshold 0.47, 104,897 parameters.

Widening improved Dice by only 0.003942 absolute versus the narrow candidate
(0.3942 percentage points). Weight 6 did not improve Dice over the unweighted
wide model. These small single-seed differences are not evidence of statistical
significance. Different thresholds and loss weights also mean the probability
maps should not be interpreted as equally calibrated probabilities.

## Selected Model Metrics

All-observed validation population: 7,500,385 pixels, 102,594 positive (1.368%).
Unknown target pixels are excluded; metrics pool pixels rather than average images.

- F1/Dice: 0.305007; IoU: 0.179946.
- Precision: 0.265737; recall: 0.357896.
- Trapezoidal PR-AUC: 0.192103; average precision: 0.192116.
- TP 36,718; FP 101,456; FN 65,876; TN 7,296,335.

Thus about 64.2% of observed active-fire pixels are missed, and 73.4% of predicted
fire pixels are false positives at the selected operating point.

New-fire validation population: all 7,342,003 observed targets whose previous
fire mask is known non-fire, including negative candidates. There are 83,088
positive targets (1.132%). The same threshold 0.09 is used without retuning.

- F1/Dice: 0.232738; IoU: 0.131694.
- Precision: 0.231839; recall: 0.233644.
- Trapezoidal PR-AUC: 0.131965; average precision: 0.131979.
- TP 19,413; FP 64,322; FN 63,675; TN 7,194,593.

The model detects about 23.4% of newly active fire pixels. Spread prediction
remains a substantial weakness even though overall metrics are above zero.

## Prediction Map Review

All three PNGs show the same first four ordered validation examples, not
handpicked successes. Columns are today's mask, tomorrow's target, predicted
probability and thresholded prediction. Red is active fire, white is non-fire
and gray is unknown (not scored). The probability color scale spans 0 to 1,
so an apparently dark image can still contain nonzero probabilities.

- Sample 2 contains observed small fires tomorrow, but no active-fire predictions
  are visible for any of the three candidates: an example of false negatives.
- Sample 4 predicts a patch around today's fire although no active-fire target
  is visible in its observed tomorrow pixels: an example of false positives.
- Samples 1 and 3 have no visible tomorrow fire; predicted observed pixels are
  likewise non-fire. Large gray regions must not be counted as correct negatives.

These four maps illustrate errors but cannot quantify their population frequency.
They do not establish collapse to an all-background solution: pooled TP counts
are nonzero for all candidates.

## Interpretation and Next Steps

The wide model's Dice peaks at epoch 7; subsequent epochs and learning-rate
reductions do not exceed that peak. Simply allowing 40 epochs did not resolve
the validation plateau. GPU execution makes the same experiment faster; it
does not by itself improve accuracy.

The old 0.349473 value is a test score from a different, unclipped preprocessing
run. It is not directly comparable to the current 0.305007 validation score.
No performance improvement or regression on held-out test data is established.

Before another tuning sweep, retain this locked winner and establish a matched
validation baseline (including persistence and fire-only) with identical clips,
splits and metric definitions. Inspect a larger deterministic collection of
false-negative/false-positive examples, including unknown-input coverage.
Use additional seeds to assess stability. Any exploratory reuse of the previously
viewed official test must be labeled as such; a fresh event-disjoint holdout is
needed for a stronger generalization claim. No new training job was launched
during this review.
