# Dataset inspection

## Where files go

- `data/raw/`: extracted original TFRecord files; keep them unchanged.
- `data/processed/`: derived data, if needed later.
- `notebooks/`: exploration notebooks.
- `src/`: reusable Python code, once exploration needs it.
- `outputs/figures/`: exported sample maps and plots.
- `outputs/reports/`: dataset summaries and statistics.

Download the original Next Day Wildfire Spread dataset:
https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread

Extract the `.tfrecord` files into `data/raw/`. Preserve the supplied filenames
and train, eval (validation), and test partitions. TFRecord files are binary
records, so inspect them with a Python reader, not a text editor.

## First exploration notebook

Create `notebooks/01_inspect_dataset.ipynb` when the data is available. Start
with one training file, rather than loading the entire dataset into memory.

1. List the files by split and check file sizes.
2. Decode one training record and print its feature names, types, and shapes.
3. Check for the expected 12 input maps and one target map, each 64 x 64.
4. Plot `PrevFireMask` and `FireMask` side by side with a shared discrete color
   legend: -1 unknown, 0 no active fire, 1 active fire. Keep unknown pixels
   visually distinct from non-fire pixels.
5. Plot environmental layers, starting with elevation, NDVI, wind speed, and
   temperature. Give each layer its own color scale.
6. On a small training subset, count unknown, non-fire, and fire target pixels;
   check input ranges and non-finite values. Label these statistics as a subset.
7. Record observations and save representative figures. Calculate normalization
   statistics from training data only when preparing for modeling.

Expected input keys: `elevation`, `pdsi`, `NDVI`, `pr`, `sph`, `th`, `tmmn`,
`tmmx`, `vs`, `erc`, `population`, and `PrevFireMask`. Target: `FireMask`.

Do not shuffle records into new splits or assume adjacent records describe
consecutive days of the same fire. Unknown targets must be excluded from
evaluation and training loss.

For cloud execution, import `notebooks/02_unet_experiments.ipynb` into Kaggle
and attach the original dataset through Add Input. The notebook contains the
reader, input plots, training statistics and controlled U-Net comparisons.
Your local `data/raw/` can stay empty.
