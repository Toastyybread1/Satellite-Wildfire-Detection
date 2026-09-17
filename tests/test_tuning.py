"""Verify improvement helpers without reading real data or scoring test."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from test_experiment import helpers


def tuning_helpers():
    namespace = helpers()
    source = Path(__file__).resolve().parents[1] / 'src/tuning.py'
    module = ast.parse(source.read_text())
    declarations = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = tuning_helpers()

    def test_unit_weight_preserves_baseline_loss(self):
        y = tf.constant([[[[1.], [-1.]], [[0.], [0.]]]])
        p = tf.constant([[[[0.6], [0.9]], [[0.1], [0.2]]]])
        self.assertAlmostEqual(float(self.h['masked_bce_dice'](y, p)),
            float(self.h['weighted_masked_bce_dice'](y, p, 1.)))
        self.assertGreater(float(self.h['weighted_masked_bce_dice'](y, p, 6.)),
            float(self.h['weighted_masked_bce_dice'](y, p, 1.)))

    def test_weighted_loss_ignores_unknown_and_has_zero_unknown_gradient(self):
        y = tf.constant([[[[1.], [-1.]], [[0.], [0.]]]])
        p = tf.Variable([[[[0.6], [0.9]], [[0.1], [0.2]]]])
        with tf.GradientTape() as tape:
            loss = self.h['weighted_masked_bce_dice'](y, p, 6.)
        self.assertEqual(float(tape.gradient(loss, p)[0, 0, 1, 0]), 0.)
        self.assertEqual(float(self.h['weighted_masked_bce_dice'](-tf.ones_like(y), p, 6.)), 0.)

    def test_wider_model_has_more_parameters_and_can_train(self):
        narrow = self.h['build_tuned_unet'](4, 1.)
        wide = self.h['build_tuned_unet'](8, 6.)
        self.assertGreater(wide.count_params(), narrow.count_params())
        x = tf.random.normal((2, 64, 64, 12))
        y = np.zeros((2, 64, 64, 1), dtype='float32')
        y[:, 28:32, 28:32] = 1.
        self.assertTrue(np.isfinite(wide.train_on_batch(x, y)))

    def test_callback_selects_checkpoint_using_validation_only(self):
        model = self.h['build_tuned_unet'](4, 1.)
        original = self.h['collect_predictions']
        with tempfile.TemporaryDirectory() as folder:
            self.h['OUTPUT_ROOT'] = Path(folder)
            splits = []

            def predictions(split, model, keys):
                splits.append(split)
                return np.array([0.9, 0.1]), np.array([1, 0]), np.array([0, 0])

            self.h['collect_predictions'] = predictions
            callback = self.h['ValidationDiceCheckpoint']('tiny')
            callback.set_model(model)
            logs = {}
            callback.on_epoch_end(0, logs)
            callback.on_epoch_end(1, {})
            self.assertEqual(logs['val_dice'], 1.)
            self.assertEqual(callback.best_epoch, 1)
            self.assertEqual(splits, ['validation', 'validation'])
            self.assertTrue((Path(folder) / 'tiny.weights.h5').is_file())
        self.h['collect_predictions'] = original

    def test_generated_tuning_never_invokes_test_predictions(self):
        import json
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/03_unet_validation_tuning.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        tree = ast.parse(code)
        # Baseline helper definitions mention test maps but must never be called.
        self.assertNotIn("collect_predictions('test'", code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, 'prediction_maps')

    def test_nonfinite_validation_does_not_save_checkpoint(self):
        original = self.h['collect_predictions']
        self.h['collect_predictions'] = lambda *args: (
            np.array([np.nan, 0.1]), np.array([1, 0]), np.array([0, 0]))
        try:
            callback = self.h['ValidationDiceCheckpoint']('invalid')
            with self.assertRaises(FloatingPointError):
                callback.on_epoch_end(0, {})
            self.assertEqual(callback.best_f1, -1.)
        finally:
            self.h['collect_predictions'] = original


if __name__ == '__main__':
    unittest.main()
