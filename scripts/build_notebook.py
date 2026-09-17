"""Generate the portable Kaggle notebook from the experiment's cell markers."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'src/experiment.py').read_text()
intro = '''# Next-day wildfire spread: controlled U-Net experiment

Attach [the original dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
with Kaggle's Add Input control, then run cells in order. The dataset stays on
Kaggle. GPU acceleration is recommended if your account supports it.

Inspired by [Anastasiou et al. (2025)](https://arxiv.org/abs/2505.17556): one
encoder/pooling stage, a bottleneck, a skip-connected decoder, batch normalization,
GELU and dropout; equal-weight masked BCE/Dice loss. This predicts next-day active
fire, whereas the paper predicts final burned area using Mesogeos.

Four trained variants use identical 12-channel model sizes, initial seeds and
training budgets. Excluded channels are zero after training-only standardization.
Weather ablation removes precipitation, humidity, wind direction/speed and minimum/
maximum temperature. Vegetation ablation removes NDVI. Drought (PDSI) and fire
danger (ERC) remain, so these are narrow feature-group ablations.

Best checkpoints minimize validation loss; thresholds maximize validation F1.
All choices are saved before test evaluation. Test results pool observed pixels
and report F1/Dice, IoU, precision, recall, exact trapezoidal PR-AUC and average
precision. New-fire evaluation includes **all known previously non-fire pixels**,
including negatives. No new-fire-specific threshold is tuned.

One seed is an initial comparison, not evidence of statistical significance.
Repeat with additional seeds before drawing strong conclusions. Unknown target
pixels never enter loss or metrics. Input statistics use training only. Non-finite
inputs and invalid zero-Kelvin temperatures are replaced by their training means.
The dataset authors' published clipping bounds are applied before fitting
training-only mean/std, so export outliers do not dominate normalization. Raw
ranges and counts outside the bounds are retained in the inspection report.
Unknown previous-fire values stay -1. For persistence they predict non-fire; a
known-previous scoring population is also reported.

Outputs, weights, settings, histories and plots go under
`/kaggle/working/wildfire_experiment/`. The first four test samples are plotted
without selecting examples based on results. No training results are bundled
in this source notebook until it is executed on Kaggle.
'''
headings = [
    '## 1. Settings and reproducibility',
    '## 2. Supplied splits and all input layers',
    '## 3. Training-only statistics and preprocessing',
    '## 4. Shallow U-Net and masked BCE/Dice loss',
    '## 5. Evaluation, threshold selection and sanity checks',
    '## 6. Train four variants and lock validation choices',
    '## 7. Test evaluation and prediction maps',
]


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


cells = [markdown(intro)]
for heading, chunk in zip(headings, source.split('# %%')[1:]):
    cells.append(markdown(heading))
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': chunk.strip().splitlines(True)})
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                       'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.11'}}}
# Deterministic cell IDs keep notebook diffs readable.
for i, cell in enumerate(cells):
    cell['id'] = f'wildfire-{i:02d}'
target = root / 'notebooks/02_unet_experiments.ipynb'
target.write_text(json.dumps(notebook, indent=1) + '\n')
print(target)

# Reuse inspected parsing/preprocessing/helpers, but never the baseline training
# or test-evaluation cells, in the validation-only improvement notebook.
tuning_chunks = (root / 'src/tuning.py').read_text().split('# %%')[1:]
tuning_intro = '''# Next-day wildfire spread: validation-only U-Net tuning

Attach [Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread).
The dataset stays on Kaggle; edits can be submitted from VS Code with the official CLI.
GPU execution is recommended: these full-data comparisons can take many hours on CPU.

This extends the paper-inspired shallow GELU/BN U-Net, retaining one pooling stage.
Three predefined all-feature candidates compare longer training (16 filters),
wider layers (32 filters), and wider layers with positive BCE weight 6.
All use seed 42, gradient clipping, up to 40 epochs and learning-rate reduction.
An epoch-end callback selects checkpoint and threshold jointly by pooled validation
F1/Dice. Early stopping also monitors validation Dice. Metrics use the same unknown
pixel exclusions and new-fire candidate population as the baseline experiment.

Input normalization is fitted on training only after the authors' published clips.
Non-finite and nonpositive Kelvin inputs are mean-imputed. No geometric flips or
rotations are added because wind direction would need a matching physical transform.

This notebook does **not evaluate test data**. Baseline test scores have already been
viewed, so improved validation scores are not proof of improved held-out performance.
Repeated validation selection can also overfit: verify the selected model on a fresh,
event-disjoint holdout and additional seeds before claiming a generalization gain.
The original official test can only be reported as a reused exploratory benchmark.
Neither a target Dice score nor an improvement is guaranteed.

Outputs include validation histories, metrics, fixed first-four validation maps,
model weights, configuration, locked choices and selected_model.json under
`/kaggle/working/wildfire_tuning/`. No test metrics are written.
'''
tuning_cells = [markdown(tuning_intro)]
base_chunks = source.split('# %%')[1:6]
assert len(base_chunks) == 5 and len(tuning_chunks) == 2
for heading, chunk in zip(headings[:5], base_chunks):
    chunk = chunk.replace("Path('/kaggle/working/wildfire_experiment')",
                          "Path('/kaggle/working/wildfire_tuning')")
    tuning_cells.extend([markdown(heading), {
        'cell_type': 'code', 'metadata': {}, 'execution_count': None,
        'outputs': [], 'source': chunk.strip().splitlines(True)}])
for heading, chunk in zip(['## 6. Tuning candidates and validation Dice checkpoint',
                            '## 7. Train and select using validation only'], tuning_chunks):
    tuning_cells.extend([markdown(heading), {
        'cell_type': 'code', 'metadata': {}, 'execution_count': None,
        'outputs': [], 'source': chunk.strip().splitlines(True)}])
for i, cell in enumerate(tuning_cells):
    cell['id'] = f'wildfire-tuning-{i:02d}'
notebook['cells'] = tuning_cells
target = root / 'notebooks/03_unet_validation_tuning.ipynb'
target.write_text(json.dumps(notebook, indent=1) + '\n')
print(target)
