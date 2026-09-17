# VS Code and Kaggle

## Connection Model

Edit code and notebooks locally in VS Code, submit them to Kaggle, and download
small output reports. Kaggle mounts the attached dataset in its cloud runtime.
This is a code/run synchronization workflow, not a live VS Code remote kernel
or a filesystem mount. Kaggle input files do not appear on your Mac.

The dataset source is automatically attached by the generated kernel metadata:
`fantineh/next-day-wildfire-spread`. Dataset inputs are never downloaded by the
workspace helper. Downloading even one original TFRecord would require a local
copy of that shard; streaming does not eliminate transferring bytes.

Public file metadata can already be inspected without login:

```sh
.venv/bin/python scripts/kaggle_workspace.py inspect
```

This command was verified against Kaggle and found all 19 TFRecord shards.
It saves `data/dataset-manifest.json` (names and sizes only), not dataset inputs.

## Authenticate Once

Authentication was completed with the official OAuth flow on September 17,
2026, and private-kernel access was verified as `kevinhou74`. The tuning notebook
was submitted successfully. The commands below are for future setup or when
authentication expires, not a step to repeat for the current running job.
Account verification subsequently unlocked GPU access. Tuning Version 2
(350640755) was allocated GPU T4 x2, and TensorFlow detected both Tesla T4
devices. The model uses the default single-GPU placement rather than distributing
training across both devices.

In VS Code's terminal, from the repository root:

```sh
.venv/bin/python -m pip install -r requirements-kaggle.txt
.venv/bin/kaggle auth login
```

Complete the official browser authorization yourself. Do not paste tokens in
chat, notebooks or source files. The CLI stores its own credentials outside the
repo. No credential was created or collected by the workspace helper.

VS Code tasks are also available through Terminal > Run Task: `Kaggle: Login`,
`Kaggle: Run Validation Tuning (GPU)`, `Kaggle: Tuning Status` and report downloads.
`Kaggle: Inspect Dataset Files` lists public input metadata without downloading it.
GPU eligibility/phone verification must be completed by the account owner on
Kaggle. A logged-in browser does not automatically authenticate the local CLI.

## Submit and Retrieve

Edit `src/experiment.py` for the baseline or `src/tuning.py` for improvements;
the generator embeds those sources in self-contained notebooks. The run command
rebuilds notebooks before submission, so direct edits to generated notebooks
would be overwritten. Both kernels remain private, with internet disabled.

```sh
.venv/bin/python scripts/kaggle_workspace.py prepare --experiment tuning --gpu
.venv/bin/python scripts/kaggle_workspace.py run --experiment tuning --gpu
.venv/bin/python scripts/kaggle_workspace.py status --experiment tuning
.venv/bin/python scripts/kaggle_workspace.py reports --experiment tuning
```

`prepare` is offline and only writes ignored staging files. `run` uploads code
and starts a full-data Kaggle run. Omit `--gpu` for CPU execution, which may take
many hours. Do not start duplicate jobs while an equivalent job is running.
`reports` downloads CSV/JSON/PNG/NPZ outputs only, excluding input TFRecords and
model weights. Outputs go under `outputs/reports/tuning/`, outside Git.

Use `--experiment baseline` to inspect/download the existing comparison kernel;
running it again would create another version of that kernel. Each submission
gets a Kaggle version number. Downloads default to the current successful output;
check the version/status before attributing results to new code.

## Improving Dice

`notebooks/03_unet_validation_tuning.ipynb` tests three predefined improvements:
longer training, wider shallow layers, and positive-class-weighted BCE/Dice loss.
Checkpoint selection now targets pooled validation Dice directly instead of
assuming minimum BCE/Dice loss gives the best thresholded Dice. A scheduler
reduces learning rate when validation loss stops improving.

All candidates are selected on validation only. The notebook writes no test
scores. Validation gains must not be compared numerically to the baseline's
0.349 test Dice as if both were measured on the same population. Since original
test results were already seen, strong improvement claims require a fresh
event-disjoint holdout and additional seeds. New-fire scoring keeps non-fire
negatives, so it cannot be inflated by evaluating only positive spread pixels.

## Official Documentation

- [Authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Notebook submission and outputs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Dataset attachment metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
