"""Check offline Kaggle staging and safe command construction."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workspace', ROOT / 'scripts/kaggle_workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class KaggleWorkspaceTests(unittest.TestCase):
    def test_prepare_attaches_dataset_and_keeps_kernel_private(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            notebooks = root / 'notebooks'
            notebooks.mkdir()
            (notebooks / '03_unet_validation_tuning.ipynb').write_text('{}')
            with patch.object(workspace, 'ROOT', root):
                stage, kernel = workspace.prepare('tuning', gpu=True)
            metadata = json.loads((stage / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['dataset_sources'], ['fantineh/next-day-wildfire-spread'])
            self.assertTrue(metadata['is_private'])
            self.assertTrue(metadata['enable_gpu'])
            self.assertFalse(metadata['enable_internet'])
            self.assertEqual(kernel, 'kevinhou74/next-day-wildfire-u-net-validation-tuning')
            self.assertTrue((stage / metadata['code_file']).exists())

    def test_reports_never_requests_dataset_or_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(workspace, 'ROOT', Path(folder)), \
                 patch.object(workspace.shutil, 'which', return_value='/safe/kaggle'), \
                 patch.object(workspace.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
                 patch('sys.argv', ['workspace', 'reports']):
                with self.assertRaises(SystemExit) as exit:
                    workspace.main()
                self.assertEqual(exit.exception.code, 0)
                command = run.call_args.args[0]
                self.assertEqual(command[1:3], ['kernels', 'output'])
                self.assertEqual(command[-2:], ['--file-pattern', r'\.(csv|json|png|npz)$'])

    def test_inspect_saves_only_file_metadata(self):
        response = 'name,size,creationDate\ntrain.tfrecord,123,2021-01-01\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(workspace, 'ROOT', root), \
                 patch.object(workspace.shutil, 'which', return_value='/safe/kaggle'), \
                 patch.object(workspace.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, response)) as run, \
                 patch('sys.argv', ['workspace', 'inspect']):
                workspace.main()
            self.assertEqual(run.call_args.args[0][1:3], ['datasets', 'files'])
            manifest = json.loads((root / 'data/dataset-manifest.json').read_text())
            self.assertEqual(manifest['file_count'], 1)
            self.assertEqual(manifest['total_uncompressed_bytes'], 123)
            self.assertFalse(list(root.rglob('*.tfrecord')))


if __name__ == '__main__':
    unittest.main()
