# Next-Day Wildfire Spread Prediction

Predict tomorrow's satellite-observed active-fire mask from today's fire mask,
weather, vegetation, and terrain. Active fire is different from cumulative burned
area. The experiment adapts the shallow 2D U-Net in
[Wildfire spread forecasting with Deep Learning](https://arxiv.org/abs/2505.17556)
to the original Next Day Wildfire Spread dataset. The paper uses Mesogeos and a
final-burned-area target; this project uses a next-day active-fire target.

## Run on Kaggle

The first complete comparison finished on Kaggle; a corrected preprocessing run
is in progress. See [run records](docs/experiment-runs.md) for version IDs,
preliminary scores, limitations and artifact locations.

Import `notebooks/02_unet_experiments.ipynb` into a Kaggle notebook and attach
[Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
using Add Input. Run the cells in order. The dataset stays on Kaggle; no local
dataset download is required. Enable a GPU if your account supports it.

The notebook inspects all input layers, computes training-only normalization,
then trains fire-only, all-feature, no-weather, and no-NDVI U-Nets. Every variant
uses the same 12-channel architecture; excluded standardized channels are zero.
The shallow network has one downsampling stage, skip connections, batch
normalization, GELU, dropout, and equal-weight masked BCE/Dice loss.
Published clipping bounds from the dataset authors are applied before computing
training-only mean/std. The statistics report retains raw ranges and outlier
counts. Missing inputs are filled with training means; unknown fire masks remain -1.

Best checkpoints minimize validation loss. Thresholds maximize pooled validation
F1 on a fixed grid. All choices are locked before test evaluation. Persistence
copies today's observed fire; unknown previous pixels predict non-fire. Results
include all observed targets, known-previous pixels, and new-fire candidates
(previously non-fire pixels, including negatives).

Test metrics are pooled pixel F1/Dice, IoU, precision, recall, trapezoidal PR-AUC,
and average precision. Outputs include fixed-sample prediction maps, PR curves,
histories, model weights, normalization and configuration. They are saved under
`/kaggle/working/wildfire_experiment/`. Download the small artifacts you need to
`outputs/figures/` or `outputs/reports/`; keep weights outside Git.

Weather ablation removes `pr`, `sph`, `th`, `tmmn`, `tmmx`, and `vs`; vegetation
ablation removes `NDVI`. Both retain PDSI and ERC. This initial run uses one seed;
repeat with multiple seeds before making strong claims about contributions.

## Local Code

[VS Code/Kaggle setup](docs/vscode-kaggle.md) provides official CLI login,
cloud-run tasks, automatic dataset attachment and small-report downloads.
`notebooks/03_unet_validation_tuning.ipynb` adds validation-only comparisons for
longer/wider training and class-weighted loss, without changing the baseline.
It does not yet demonstrate a higher test Dice score.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebook is generated from `src/experiment.py`. After changing that source,
run `python3 scripts/build_notebook.py` and import the updated notebook into Kaggle.
Local execution requires changing `INPUT_ROOT` and `OUTPUT_ROOT` to local paths
and having the data available. Viewing the notebook in VS Code/GitHub requires
no dataset. See `docs/dataset-inspection.md` for the folder layout.

## Data

Inputs are 12 maps of shape 64 x 64: elevation, pdsi, NDVI, pr, sph, th, tmmn,
tmmx, vs, erc, population, PrevFireMask. The target is FireMask.
Fire values are -1 (unknown), 0 (no active fire), and 1 (active fire).
Unknown targets are excluded from scoring. Persistence predicts fire only where
the previous mask is 1. Separate metrics restrict evaluation to known previous
observations and to previously non-fire pixels (new fire).

## Next Milestones

1. Review the completed first run and the corrected clipping run on Kaggle.
2. Record corrected metrics and inspect prediction errors.
3. Repeat with additional seeds and analyze failures without retuning on test.
4. Summarize environmental feature ablations and new-fire performance.
5. Add temporal models only with verified event IDs, dates, and aligned grids.

Keep related fire observations together when creating custom splits. Do not
assume consecutive records form a time series. Six-hour forecasting requires
new targets aligned to that horizon.

## References

- [Dataset paper](https://research.google/pubs/next-day-wildfire-spread-a-machine-learning-dataset-to-predict-wildfire-spreading-from-remote-sensing-data/)
- [Official schema and code](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread)
- [WildfireSpreadTS for later temporal experiments](https://github.com/SebastianGer/WildfireSpreadTS)

The notebook includes sanity checks for masked loss, output shape, metric counts,
and threshold selection before training.

Local verification (synthetic data only):

```sh
python -m unittest discover -s tests -v
python tests/smoke_experiment.py
```
