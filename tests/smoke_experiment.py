"""Run the complete experiment on tiny synthetic TFRecords, never real data."""
import ast
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use('Agg')
root = Path(__file__).resolve().parents[1]
source = ast.parse((root / 'src/experiment.py').read_text())
rng = np.random.default_rng(123)
features = ['elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 'tmmn',
            'tmmx', 'vs', 'erc', 'population', 'PrevFireMask']

with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    inputs, outputs = folder / 'input', folder / 'output'
    inputs.mkdir()
    for split in ['train', 'eval', 'test']:
        with tf.io.TFRecordWriter(str(inputs / f'next_day_wildfire_spread_{split}_00.tfrecord')) as writer:
            for i in range(4):
                layers = {k: (rng.random((64, 64)) * 10 + 280).astype('float32')
                          for k in features}
                previous = np.zeros((64, 64), 'float32')
                previous[25:30, 25 + i:30 + i] = 1.
                previous[0] = -1.
                target = np.roll(previous, 1, axis=1)
                target[1] = -1.
                layers.update(PrevFireMask=previous, FireMask=target)
                fields = {k: tf.train.Feature(float_list=tf.train.FloatList(value=v.ravel()))
                          for k, v in layers.items()}
                writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
    replacements = {'EPOCHS': ast.Constant(1), 'BASE_FILTERS': ast.Constant(4),
                    'BATCH_SIZE': ast.Constant(2),
                    'INPUT_ROOT': ast.Call(ast.Name('Path', ast.Load()), [ast.Constant(str(inputs))], []),
                    'OUTPUT_ROOT': ast.Call(ast.Name('Path', ast.Load()), [ast.Constant(str(outputs))], [])}
    for node in source.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in replacements:
                node.value = replacements[name]
    exec(compile(ast.fix_missing_locations(source), 'synthetic_experiment', 'exec'), {})
    report = pd.read_csv(outputs / 'test_metrics.csv')
    assert len(report) == 15
    assert set(report['model']) == {'persistence', 'fire_only', 'all_features',
                                    'without_weather', 'without_vegetation'}
    assert len(list(outputs.glob('*.weights.h5'))) == 4
    assert len(list(outputs.glob('*prediction_maps.png'))) == 5
    assert (outputs / 'locked_validation_choices.json').exists()
    assert (outputs / 'test_precision_recall.png').exists()
    print('PASS: complete synthetic experiment produced all five comparisons and artifacts.')
