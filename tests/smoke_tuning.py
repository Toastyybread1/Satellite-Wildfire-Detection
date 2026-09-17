"""Execute the generated validation-only notebook on synthetic TFRecords."""
import ast
import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use('Agg')
root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'notebooks/03_unet_validation_tuning.ipynb').read_text())
code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
tree = ast.parse(code)
rng = np.random.default_rng(123)
features = ['elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 'tmmn',
            'tmmx', 'vs', 'erc', 'population', 'PrevFireMask']
with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    inputs, outputs = folder / 'input', folder / 'output'
    inputs.mkdir()
    # The test filename exists for split discovery, but deliberately contains
    # invalid bytes. Any accidental test read must fail the integration test.
    (inputs / 'next_day_wildfire_spread_test_00.tfrecord').write_bytes(b'not a TFRecord')
    for split in ['train', 'eval']:
        with tf.io.TFRecordWriter(str(inputs / f'next_day_wildfire_spread_{split}_00.tfrecord')) as writer:
            for i in range(4):
                layers = {k: (rng.random((64, 64)) * 10 + 280).astype('float32') for k in features}
                prev = np.zeros((64, 64), 'float32')
                prev[25:30, 25 + i:30 + i] = 1.
                prev[0] = -1.
                layers.update(PrevFireMask=prev, FireMask=np.roll(prev, 1, axis=1))
                fields = {k: tf.train.Feature(float_list=tf.train.FloatList(value=v.ravel()))
                          for k, v in layers.items()}
                writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
    replacements = {'EPOCHS': ast.Constant(1), 'TUNING_EPOCHS': ast.Constant(1),
        'BASE_FILTERS': ast.Constant(4), 'BATCH_SIZE': ast.Constant(2),
        'INPUT_ROOT': ast.parse(f'Path({str(inputs)!r})', mode='eval').body,
        'OUTPUT_ROOT': ast.parse(f'Path({str(outputs)!r})', mode='eval').body}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in replacements:
                node.value = replacements[name]
            elif name == 'TUNING_CANDIDATES':
                candidates = ast.literal_eval(node.value)
                for c in candidates:
                    c['filters'] = 4
                node.value = ast.parse(repr(candidates), mode='eval').body
    exec(compile(ast.fix_missing_locations(tree), 'synthetic_tuning', 'exec'), {})
    report = pd.read_csv(outputs / 'validation_metrics.csv')
    assert len(report) == 9
    assert len(list(outputs.glob('*.weights.h5'))) == 3
    assert len(list(outputs.glob('*validation_maps.png'))) == 3
    assert (outputs / 'selected_model.json').is_file()
    assert not list(outputs.glob('test_*'))
    print('PASS: all three tuning candidates completed; unreadable test data was never accessed.')
