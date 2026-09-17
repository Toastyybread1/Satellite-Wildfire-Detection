"""Exercise notebook helpers without starting a real-data experiment."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf


def helpers():
    source = Path(__file__).resolve().parents[1] / 'src/experiment.py'
    module = ast.parse(source.read_text())
    constants = {'FEATURES', 'SCHEMA', 'BASE_FILTERS', 'LEARNING_RATE', 'SEED',
                 'BATCH_SIZE', 'WEATHER', 'VEGETATION', 'VARIANTS', 'CLIP_BOUNDS'}
    declarations = [node for node in module.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
        and t.id in constants for t in node.targets))]
    namespace = {}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = helpers()
        tf.keras.utils.set_random_seed(42)

    def test_unknown_targets_have_no_loss_or_gradient(self):
        target = tf.constant([[[[1.], [-1.]], [[0.], [0.]]]])
        pred = tf.Variable([[[[0.8], [0.1]], [[0.2], [0.2]]]])
        with tf.GradientTape() as tape:
            loss = self.h['masked_bce_dice'](target, pred)
        gradient = tape.gradient(loss, pred).numpy()
        self.assertEqual(gradient[0, 0, 1, 0], 0.)
        modified = tf.constant([[[[0.8], [0.9]], [[0.2], [0.2]]]])
        self.assertAlmostEqual(float(loss), float(self.h['masked_bce_dice'](target, modified)))
        self.assertEqual(float(self.h['masked_bce_dice'](-tf.ones_like(target), pred)), 0.)

    def test_counts_and_undefined_metrics(self):
        p = np.array([0.9, 0.8, 0.1, 0.99])
        y = np.array([1, 0, 1, -1])
        score, _ = self.h['score_pixels'](p, y, 0.5, y >= 0)
        self.assertEqual((score['tp'], score['fp'], score['fn'], score['tn']), (1, 1, 1, 0))
        self.assertEqual(score['f1_dice'], 0.5)
        self.assertAlmostEqual(score['iou'], 1 / 3)
        empty, curve = self.h['score_pixels'](p, y, 0.5, np.zeros(4, bool))
        self.assertIsNone(empty['f1_dice'])
        self.assertIsNone(empty['pr_auc'])
        self.assertIsNone(curve)

    def test_new_fire_scores_include_negative_pixels(self):
        p = np.array([0.9, 0.8, 0.8, 0.9, 0.1])
        y = np.array([1, 0, 1, 1, -1])
        previous = np.array([0, 0, 1, -1, 0])
        with tempfile.TemporaryDirectory() as folder:
            self.h['OUTPUT_ROOT'] = Path(folder)
            reports = self.h['evaluate']('toy', p, y, previous, 0.5)
        new = next(r for r in reports if r['population'] == 'new_fire')
        self.assertEqual(new['pixels'], 2)
        self.assertEqual(new['tp'], 1)
        self.assertEqual(new['fp'], 1)

    def test_validation_threshold(self):
        threshold, curve = self.h['choose_threshold'](
            np.array([0.2, 0.8, 0.99]), np.array([0, 1, -1]))
        self.assertTrue(0.2 < threshold <= 0.8)
        self.assertEqual(curve['f1'].max(), 1.)
        with self.assertRaises(ValueError):
            self.h['choose_threshold'](np.array([0.1]), np.array([0]))

    def test_record_schema_and_missing_temperature_imputation(self):
        fields = {k: tf.train.Feature(float_list=tf.train.FloatList(
            value=np.full(4096, 1., dtype='float32'))) for k in self.h['FEATURES'] + ['FireMask']}
        fields['tmmx'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.zeros(4096)))
        record = tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.tfrecord'
            with tf.io.TFRecordWriter(str(path)) as writer:
                writer.write(record)
            raw = next(iter(tf.data.TFRecordDataset(str(path))))
            x, y = self.h['decode'](raw)
        self.assertEqual(x.shape, (64, 64, 12))
        self.assertEqual(y.shape, (64, 64, 1))
        self.h['mean'] = np.full(12, 2., dtype='float32')
        self.h['std'] = np.ones(12, dtype='float32')
        normalized, _ = self.h['normalize'](x, y)
        self.assertTrue(np.all(normalized.numpy()[..., 7] == 0.))
        self.assertEqual(int(np.isnan(self.h['finite_inputs'](x.numpy())[..., 7]).sum()), 4096)

    def test_model_can_update_weights(self):
        model = self.h['build_unet']()
        x = tf.random.normal((2, 64, 64, 12))
        y = np.zeros((2, 64, 64, 1), dtype='float32')
        y[:, 30:34, 30:34] = 1.
        y[:, :2] = -1.
        before = model.trainable_weights[0].numpy().copy()
        loss = model.train_on_batch(x, y)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, model.trainable_weights[0].numpy()))
        self.assertEqual(model(x, training=False).shape, (2, 64, 64, 1))

    def test_clipping_matches_numpy_and_tensorflow(self):
        x = np.full((64, 64, 12), 100000., dtype='float32')
        x[..., 8] = -10.  # Impossible negative wind speed.
        x[..., -1] = -1.
        x[0, 0, 0] = np.inf
        x[0, 1, 1] = -np.inf
        x[0, 2, 2] = np.nan
        clipped = self.h['finite_inputs'](x)
        self.assertTrue(np.all(clipped[..., 5] == 360.))
        self.assertTrue(np.all(clipped[..., 8] == 0.))
        self.assertTrue(np.all(clipped[..., -1] == -1.))
        self.h['mean'] = np.zeros(12, dtype='float32')
        self.h['std'] = np.ones(12, dtype='float32')
        normalized, _ = self.h['normalize'](tf.constant(x), tf.zeros((64, 64, 1)))
        np.testing.assert_allclose(normalized.numpy(), np.nan_to_num(clipped))


if __name__ == '__main__':
    unittest.main()
