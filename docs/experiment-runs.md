# Kaggle Experiment Runs

Recorded September 17, 2026. Real dataset files remain on Kaggle. The original
exploration notebook was not modified. Both experiment versions are private.

## Version 1: Completed

[Completed run and logs](https://www.kaggle.com/code/kevinhou74/next-day-wildfire-u-net-comparisons/log?scriptVersionId=350469829)

Run ID 350469829 completed successfully in 14,107.3 seconds on CPU. It used the
complete supplied splits: 15 training files, two validation files and two test
files. Training contained 14,979 examples, with 658,270 positive, 59,254,803
negative and 1,440,911 unknown target pixels. Unknown targets were excluded.

This version used raw finite-feature training mean/std without published clipping.
Its results are preliminary because training inspection revealed substantial
export outliers. Keep these results distinct from the corrected version below.

All variants used seed 42, a 27,361-parameter shallow U-Net, batch size 64,
Adam learning rate 0.001, and at most 15 epochs. Checkpoints minimized validation
BCE/Dice loss; thresholds maximized validation F1 before test evaluation.

Reported test scores below are rounded values from the Kaggle logs, not estimates:

- Persistence: Dice 0.309326, IoU 0.182960, precision 0.357195, recall 0.272770,
  PR-AUC 0.319541, average precision 0.106548.
- Fire-only U-Net: Dice 0.339759, IoU 0.204644, precision 0.269410,
  recall 0.459831, PR-AUC 0.213023, average precision 0.212429. Best epoch 9 of
  13 executed; validation threshold 0.05.
- All-feature U-Net: Dice 0.349473, IoU 0.211734, precision 0.279230,
  recall 0.466934, PR-AUC 0.243967, average precision 0.243985. Best epoch 15;
  validation threshold 0.13.
- Without weather: Dice 0.345921, IoU 0.209132, precision 0.268246,
  recall 0.486915, PR-AUC 0.243155, average precision 0.243172. Best epoch 15;
  validation threshold 0.07.
- Without NDVI: Dice 0.346389, IoU 0.209475, precision 0.270472,
  recall 0.481555, PR-AUC 0.225212, average precision 0.225229. Best epoch 14;
  validation threshold 0.10.

These scores pool 6,727,699 observed test pixels, including 84,331 positive
pixels. They are micro metrics, not averages over images. One seed does not
establish significant differences or a causal contribution from an input group.
PDSI and ERC remain in both narrow feature ablations.

New-fire scoring includes all 6,589,342 known previously non-fire candidate
pixels, of which 60,758 are active tomorrow. It uses the same validation-selected
threshold as the overall evaluation, with no additional tuning:

- Persistence: Dice and IoU 0, precision undefined, recall 0, PR-AUC 0.504610,
  average precision 0.009221.
- Fire-only: Dice 0.226893, IoU 0.127963, precision 0.198357,
  recall 0.265019, PR-AUC 0.122364, average precision 0.121814.
- All features: Dice 0.238575, IoU 0.135445, precision 0.205260,
  recall 0.284802, PR-AUC 0.137547, average precision 0.137571.
- Without weather: Dice 0.240701, IoU 0.136816, precision 0.197867,
  recall 0.307202, PR-AUC 0.137302, average precision 0.137324.
- Without NDVI: Dice 0.239112, IoU 0.135791, precision 0.199850,
  recall 0.297574, PR-AUC 0.130966, average precision 0.130988.

Persistence predicts no new-fire candidates. Its large trapezoidal PR-AUC is an
interpolation artifact for tied/constant scores, not evidence of useful spread
prediction. Average precision equals candidate prevalence for this population.

Kaggle saved 39 output files including weights, all three evaluation populations,
training histories, input visualizations, five prediction maps and PR curves.
The all-feature prediction-map preview was visually checked. The output ZIP
download was blocked by Chrome, so artifacts have not been copied into this repo.

## Version 2: Running

[Corrected run and logs](https://www.kaggle.com/code/kevinhou74/next-day-wildfire-u-net-comparisons?scriptVersionId=350599543)

Run ID 350599543 was submitted with the name "Full comparisons with published
clipping". It applies the dataset authors' predefined clipping bounds before
fitting training-only mean/std. Raw ranges and counts outside the bounds are
retained in the statistics report. Non-finite inputs and nonpositive Kelvin
temperatures are treated as missing and filled with training means.

The correction was prompted by training inspection, not by optimizing test scores.
Architecture, seed, optimizer, maximum epochs, ablation definitions and validation
selection rules are unchanged. Test results from Version 1 have now been viewed;
further performance-driven changes would require fresh held-out evaluation.

Do not attribute Version 1 metrics to the current local source/notebook. Version 2
results are pending and must be recorded separately after its run completes.

Startup verified: all 14,979 training examples were inspected, sanity checks
passed and fire-only epoch 1/15 began. Clipped wind-direction standard deviation
is 71.584647 degrees versus 3435.083853 in Version 1. This saved cloud run
continues independently of the editor/chat. CPU runtime may be several hours.

## Validation Tuning: Submitted

`03_unet_validation_tuning.ipynb` is a separate validation-only experiment.
It tests longer training, wider shallow layers and class-weighted loss, with
checkpoint/threshold selection by pooled validation Dice. It does not evaluate
test or establish an improved test score.

[Validation tuning run](https://www.kaggle.com/code/kevinhou74/next-day-wildfire-u-net-validation-tuning?scriptVersionId=350636799)

Official CLI OAuth login completed as `kevinhou74`. Version 1, run ID 350636799,
was submitted on September 17, 2026, and the authenticated status API reported
`RUNNING`. The private notebook attaches the dataset with internet disabled.
Pulled server metadata confirms `enable_gpu: true` and machine shape
`NvidiaTeslaT4`, but the actual run reports accelerator None and TensorFlow
2.20.0 reports `GPU: []` with a CUDA initialization error. This version is
executing on CPU, not GPU. The editor confirmed that phone verification was
required for accelerator access.
Startup discovered 15 training and two validation shards (also listed the two
test paths, which this validation-only notebook never reads). CPU training may
take many hours.
Version 1 reached epoch 6/40 for its first candidate before the GPU replacement
was submitted. Real final validation results and an improved held-out Dice score
are not available yet.

### Version 2: GPU Replacement Completed

[GPU tuning run](https://www.kaggle.com/code/kevinhou74/next-day-wildfire-u-net-validation-tuning?scriptVersionId=350640755)

After account verification, the editor enabled GPU selection. Version 2, run ID
350640755, was submitted with the same code and GPU enabled. The authenticated
status API now reports `COMPLETE`; Kaggle records 22m 30s and allocation
`GPU T4 x2`. This was a fresh run, not a resume from the CPU checkpoint.
TensorFlow 2.20.0 startup confirms both physical Tesla T4 devices and creates
GPU devices with 13,756 MB each. The model uses TensorFlow's default single-GPU
placement; no multi-GPU distribution strategy was added. Cancellation of the
redundant Version 1 CPU tuning job was requested only after this confirmation;
its Active Events status changed to `Cancelling...`. The corrected baseline
comparison run was left untouched.
The three candidates executed 17, 15 and 19 epochs respectively, stopping after
eight epochs without improved pooled validation Dice. The winner was
`shallow_wider`: 32 filters, positive BCE weight 1, 104,897 parameters,
checkpoint epoch 7 and validation-selected threshold 0.09.

Final all-observed validation Dice was 0.301065 for `shallow_longer`, 0.305007
for `shallow_wider`, and 0.303221 for `shallow_wider_weighted`. The winning
configuration's validation IoU was 0.179946, precision 0.265737, recall 0.357896,
PR-AUC 0.192103 and average precision 0.192116. New-fire candidate Dice was
0.232738, IoU 0.131694, precision 0.231839, recall 0.233644 and PR-AUC 0.131965.
These are validation-selection results, not improved held-out test scores.

Reports, histories and all three fixed first-four validation map images were
downloaded to `outputs/reports/tuning/`; no dataset TFRecords or model weights
were downloaded. All nine metric rows, the selected winner and executed epoch
counts were checked against the saved reports. Visual inspection found missed
small fires in sample 2 and false positives around today's fire in sample 4;
these four ordered examples are not a representative error-rate estimate.
See `docs/validation-tuning-review.md` for the interpretation and limitations.

Public dataset metadata inspection via the CLI succeeded and saved a 19-file
manifest without downloading dataset inputs. See `docs/vscode-kaggle.md`.

## Local Verification

Sixteen unit tests pass, covering masked loss/gradients, metric counts, new-fire
negative candidates, validation threshold selection, TFRecord decoding, missing
inputs, NumPy/TensorFlow clipping consistency and a model weight update. A full
synthetic experiment exercises all five comparisons and checks saved artifacts.
The tuning smoke test exercises all three candidates without reading its
deliberately invalid test shard. Synthetic scores are not real dataset results.
