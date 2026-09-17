"""Submit VS Code notebook edits to Kaggle; never download dataset inputs."""
import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    'baseline': ('02_unet_experiments.ipynb', 'Next-Day Wildfire U-Net Comparisons',
                 'next-day-wildfire-u-net-comparisons'),
    'tuning': ('03_unet_validation_tuning.ipynb', 'Next-Day Wildfire U-Net Validation Tuning',
               'next-day-wildfire-u-net-validation-tuning'),
}


def prepare(experiment, gpu=False):
    notebook, title, slug = EXPERIMENTS[experiment]
    stage = ROOT / '.kaggle-build' / experiment
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'notebooks' / notebook, stage / notebook)
    metadata = {'id': f'kevinhou74/{slug}', 'title': title, 'code_file': notebook,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': gpu, 'enable_internet': False,
        'dataset_sources': ['fantineh/next-day-wildfire-spread'],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (stage / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return stage, metadata['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'inspect', 'login', 'status', 'run', 'reports'])
    parser.add_argument('--experiment', choices=EXPERIMENTS, default='tuning')
    parser.add_argument('--gpu', action='store_true', help='Requires Kaggle GPU eligibility')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.experiment, args.gpu)[0])
        return
    executable = ROOT / '.venv' / 'bin' / 'kaggle'
    kaggle = str(executable) if executable.is_file() else shutil.which('kaggle')
    if not kaggle:
        parser.error('Install requirements-kaggle.txt in .venv first')
    if args.action == 'inspect':
        result = subprocess.run([kaggle, 'datasets', 'files',
            'fantineh/next-day-wildfire-spread', '--csv'], check=True,
            capture_output=True, text=True)
        files = list(csv.DictReader(io.StringIO(result.stdout)))
        if not files or not all('name' in f and 'size' in f for f in files):
            parser.error('Dataset file response did not contain the expected CSV schema')
        manifest = {'dataset': 'fantineh/next-day-wildfire-spread',
            'source': 'Kaggle datasets files API; metadata only',
            'file_count': len(files), 'total_uncompressed_bytes': sum(int(f['size']) for f in files),
            'files': files}
        destination = ROOT / 'data' / 'dataset-manifest.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f"Found {len(files)} files; no dataset bytes downloaded. Manifest: {destination}")
        return
    if args.action == 'login':
        command = [kaggle, 'auth', 'login']
    else:
        _, _, slug = EXPERIMENTS[args.experiment]
        kernel = f'kevinhou74/{slug}'
        if args.action == 'run':
            subprocess.run([sys.executable,
                            str(ROOT / 'scripts' / 'build_notebook.py')], check=True)
            stage, kernel = prepare(args.experiment, args.gpu)
            command = [kaggle, 'kernels', 'push', '-p', str(stage)]
        elif args.action == 'status':
            command = [kaggle, 'kernels', 'status', kernel]
        else:
            destination = ROOT / 'outputs' / 'reports' / args.experiment
            destination.mkdir(parents=True, exist_ok=True)
            command = [kaggle, 'kernels', 'output', kernel, '-p', str(destination),
                       '--file-pattern', r'\.(csv|json|png|npz)$']
    # The CLI handles auth itself. Do not print, copy or store credentials here.
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == '__main__':
    main()
