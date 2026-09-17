# %%
"""Kaggle experiment: paper-inspired next-day active-fire segmentation.

This is an adaptation of arXiv:2505.17556, not a reproduction of its
Mesogeos final-burned-area experiment. Execute the notebook cells in order.
"""
import gc
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

SEED = 42
EPOCHS = 15
BATCH_SIZE = 64
BASE_FILTERS = 16
LEARNING_RATE = 1e-3
# None means the complete supplied split. Limits are for debugging only.
TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/wildfire_experiment')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
FEATURES = ['elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 'tmmn',
            'tmmx', 'vs', 'erc', 'population', 'PrevFireMask']
# Published clipping bounds from the dataset authors' constants.py. These are
# predefined preprocessing, not thresholds learned from validation or test.
CLIP_BOUNDS = np.array([
    [0., 3141.], [-6.1, 7.9], [-9821., 9996.], [0., 44.5], [0., 1.],
    [0., 360.], [253.2, 298.9], [253.2, 315.1], [0., 10.],
    [0., 106.], [0., 2534.], [-1., 1.],
], dtype='float32')
WEATHER = {'pr', 'sph', 'th', 'tmmn', 'tmmx', 'vs'}
VEGETATION = {'NDVI'}
VARIANTS = {
    'fire_only': ['PrevFireMask'],
    'all_features': FEATURES,
    'without_weather': [k for k in FEATURES if k not in WEATHER],
    'without_vegetation': [k for k in FEATURES if k not in VEGETATION],
}
# PDSI (drought) and ERC (fire danger) remain in both narrow ablations.
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
print('TensorFlow:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))

# %%
files = sorted(INPUT_ROOT.rglob('*.tfrecord'))
split_files = {
    'train': [str(p) for p in files if '_train_' in p.name],
    'validation': [str(p) for p in files if '_eval_' in p.name],
    'test': [str(p) for p in files if '_test_' in p.name],
}
for split, paths in split_files.items():
    if not paths:
        raise FileNotFoundError(f'No {split} files. Attach fantineh/next-day-wildfire-spread.')
    print(split, len(paths), 'files')
assert len(set(sum(split_files.values(), []))) == sum(map(len, split_files.values()))

SCHEMA = {key: tf.io.FixedLenFeature([64, 64], tf.float32)
          for key in FEATURES + ['FireMask']}


def decode(record):
    fields = tf.io.parse_single_example(record, SCHEMA)
    x = tf.stack([fields[k] for k in FEATURES], axis=-1)
    y = fields['FireMask'][..., None]
    for mask in [x[..., -1:], y]:
        tf.debugging.assert_equal(
            tf.reduce_all((mask == -1) | (mask == 0) | (mask == 1)), True,
            message='Unexpected fire-mask label')
    return x, y


def raw_split(split, limit=None):
    ds = tf.data.TFRecordDataset(split_files[split], num_parallel_reads=1)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds if limit is None else ds.take(limit)


sample_x, sample_y = next(iter(raw_split('train')))
layers = {k: sample_x.numpy()[..., i] for i, k in enumerate(FEATURES)}
layers['FireMask'] = sample_y.numpy()[..., 0]
fire_cmap = ListedColormap(['gray', 'white', 'red'])
fire_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
fig, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
for ax, key in zip(axes.flat, FEATURES + ['FireMask']):
    values = layers[key]
    title = key
    if key in {'tmmn', 'tmmx'}:
        values = np.where(values > 0, values - 273.15, np.nan)
        title += ' (C)'
    elif key == 'NDVI':
        values = values * 0.0001
        title += ' (scaled index)'
    elif key == 'elevation':
        title += ' (m)'
    elif key == 'vs':
        title += ' (m/s)'
    is_fire = key in {'PrevFireMask', 'FireMask'}
    im = ax.imshow(values, cmap=fire_cmap if is_fire else 'viridis',
                   norm=fire_norm if is_fire else None, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-1, 0, 1] if is_fire else None)
for ax in list(axes.flat)[13:]:
    ax.axis('off')
fig.savefig(OUTPUT_ROOT / 'input_layers.png', dpi=150)
plt.show()

# %%
# Compute normalization from training only; stream batches instead of loading
# all records into a Python list. Zero-Kelvin temperatures are missing inputs.
def finite_inputs(x):
    x = np.array(x, dtype=np.float64, copy=True)
    x[~np.isfinite(x)] = np.nan
    for key in ['tmmn', 'tmmx']:
        i = FEATURES.index(key)
        x[..., i] = np.where(x[..., i] > 0, x[..., i], np.nan)
    return np.clip(x, CLIP_BOUNDS[:, 0], CLIP_BOUNDS[:, 1])


def training_statistics():
    count = np.zeros(12, np.int64)
    total = np.zeros(12)
    total_squared = np.zeros(12)
    low = np.full(12, np.inf)
    high = np.full(12, -np.inf)
    raw_low = np.full(12, np.inf)
    raw_high = np.full(12, -np.inf)
    clipped_count = np.zeros(12, np.int64)
    label_counts = np.zeros(3, np.int64)
    n = 0
    for x, y in raw_split('train', TRAIN_LIMIT).batch(128):
        raw = x.numpy().reshape(-1, 12)
        raw_valid = np.isfinite(raw)
        raw_low = np.minimum(raw_low, np.where(raw_valid, raw, np.inf).min(axis=0))
        raw_high = np.maximum(raw_high, np.where(raw_valid, raw, -np.inf).max(axis=0))
        clipped_count += (raw_valid & ((raw < CLIP_BOUNDS[:, 0]) |
                                       (raw > CLIP_BOUNDS[:, 1]))).sum(axis=0)
        values = finite_inputs(x.numpy()).reshape(-1, 12)
        valid = np.isfinite(values)
        safe = np.where(valid, values, 0)
        count += valid.sum(axis=0)
        total += safe.sum(axis=0)
        total_squared += np.square(safe).sum(axis=0)
        low = np.minimum(low, np.where(valid, values, np.inf).min(axis=0))
        high = np.maximum(high, np.where(valid, values, -np.inf).max(axis=0))
        labels = y.numpy()
        label_counts += [np.count_nonzero(labels == v) for v in [-1, 0, 1]]
        n += len(labels)
    if n == 0 or np.any(count == 0):
        raise ValueError('Empty training split or feature with no finite training values')
    mean = total / count
    std = np.sqrt(np.maximum(total_squared / count - mean ** 2, 0))
    std = np.maximum(std, 1e-6)
    mean[-1], std[-1] = 0., 1.  # Preserve -1/0/1 previous-fire labels.
    report = pd.DataFrame({'feature': FEATURES, 'raw_min': raw_low, 'raw_max': raw_high,
                           'clipped_min': low, 'clipped_max': high,
                           'mean': mean, 'std': std,
                           'missing_pixels': n * 4096 - count,
                           'outside_clip_bounds': clipped_count})
    report.to_csv(OUTPUT_ROOT / 'training_input_statistics.csv', index=False)
    print(report.to_string(index=False))
    print('Training records:', n, 'target counts [-1, 0, 1]:', label_counts)
    return mean.astype('float32'), std.astype('float32'), n


mean, std, n_train = training_statistics()
np.savez(OUTPUT_ROOT / 'normalization.npz', mean=mean, std=std)
limits = {'train': TRAIN_LIMIT, 'validation': VALIDATION_LIMIT, 'test': TEST_LIMIT}
metadata = {
    'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
    'base_filters': BASE_FILTERS, 'learning_rate': LEARNING_RATE,
    'limits': limits, 'features': FEATURES, 'variants': VARIANTS,
    'clip_bounds': CLIP_BOUNDS.tolist(),
    'clipping_source': 'https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/constants.py',
    'split_files': split_files, 'training_records': n_train,
    'python': platform.python_version(), 'tensorflow': tf.__version__,
    'checkpoint_selection': 'minimum validation BCE/Dice loss',
    'threshold_selection': 'maximum pooled validation F1 on 0.01..0.99 grid',
    'aggregation': 'pooled pixels (micro), not mean per image',
    'pr_auc_definition': 'trapezoidal area under exact precision-recall curve',
    'average_precision': 'also reported separately; not identical to PR-AUC',
    'new_fire_population': 'known previous non-fire pixels with known target',
    'unknown_previous_persistence': 'predict non-fire; also score known-previous subset',
    'paper': 'https://arxiv.org/abs/2505.17556',
    'dataset': 'https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread',
}
(OUTPUT_ROOT / 'config.json').write_text(json.dumps(metadata, indent=2))


def normalize(x, y):
    finite = tf.math.is_finite(x)
    for key in ['tmmn', 'tmmx']:
        i = FEATURES.index(key)
        channel_valid = x[..., i:i + 1] > 0
        finite = tf.concat([finite[..., :i], finite[..., i:i + 1] & channel_valid,
                            finite[..., i + 1:]], axis=-1)
    x = tf.clip_by_value(x, CLIP_BOUNDS[:, 0], CLIP_BOUNDS[:, 1])
    x = tf.where(finite, x, tf.constant(mean))
    return (x - tf.constant(mean)) / tf.constant(std), y


# Memory cache is shared by every variant (~4 GB for the original dataset).
# No dataset is downloaded to your Mac. Do not cache after random shuffling.
datasets = {s: raw_split(s, limits[s]).map(normalize,
            num_parallel_calls=tf.data.AUTOTUNE).cache() for s in split_files}


def feature_dataset(split, keys, training=False):
    keep = tf.constant([float(k in keys) for k in FEATURES])
    ds = datasets[split]
    if training:
        ds = ds.filter(lambda x, y: tf.reduce_any(y >= 0))
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    # Zero excluded standardized channels to preserve model size/initialization.
    ds = ds.map(lambda x, y: (x * keep, y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# %%
def masked_bce_dice(y_true, y_pred):
    valid = tf.cast(y_true >= 0, tf.float32)
    target = tf.where(y_true >= 0, y_true, 0.)
    prob = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
    axes = (1, 2, 3)
    count = tf.reduce_sum(valid, axis=axes)
    bce = -(target * tf.math.log(prob) + (1. - target) * tf.math.log(1. - prob))
    bce = tf.math.divide_no_nan(tf.reduce_sum(valid * bce, axis=axes), count)
    intersection = tf.reduce_sum(valid * target * prob, axis=axes)
    denominator = tf.reduce_sum(valid * (target + prob), axis=axes)
    dice_loss = 1. - (2. * intersection + 1.) / (denominator + 1.)
    per_sample = 0.5 * bce + 0.5 * dice_loss
    observed = tf.cast(count > 0, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(per_sample * observed),
                                 tf.reduce_sum(observed))


def conv_block(x, filters, name):
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False,
                                  name=f'{name}_conv{i}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn{i}')(x)
        x = tf.keras.layers.Activation('gelu', name=f'{name}_gelu{i}')(x)
    return tf.keras.layers.Dropout(0.1, name=f'{name}_dropout')(x)


def build_unet():
    inputs = tf.keras.Input((64, 64, len(FEATURES)))
    skip = conv_block(inputs, BASE_FILTERS, 'encoder')
    x = tf.keras.layers.MaxPooling2D()(skip)
    x = conv_block(x, 2 * BASE_FILTERS, 'bottleneck')
    x = tf.keras.layers.Conv2DTranspose(BASE_FILTERS, 2, strides=2)(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, BASE_FILTERS, 'decoder')
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, output, name='shallow_unet')
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=masked_bce_dice)
    return model


# Sanity checks: unknown labels contribute no loss, including all-unknown tiles.
t = tf.constant([[[[1.], [-1.]], [[0.], [0.]]]])
p = tf.constant([[[[0.8], [0.1]], [[0.2], [0.2]]]])
q = tf.constant([[[[0.8], [0.9]], [[0.2], [0.2]]]])
np.testing.assert_allclose(masked_bce_dice(t, p), masked_bce_dice(t, q))
assert float(masked_bce_dice(-tf.ones_like(t), p)) == 0.
probe = build_unet()
assert probe(tf.zeros((1, 64, 64, 12))).shape == (1, 64, 64, 1)
probe.summary()
del probe

# %%
def collect_predictions(split, model=None, keys=None):
    probabilities, targets, previous = [], [], []
    keep = tf.constant([float(k in (keys or FEATURES)) for k in FEATURES])
    for x, y in datasets[split].batch(BATCH_SIZE):
        prev = x[..., -1].numpy().astype(np.int8)
        prob = (prev == 1).astype(np.float32) if model is None else (
            model(x * keep, training=False).numpy()[..., 0])
        probabilities.append(prob.reshape(-1))
        targets.append(y.numpy().reshape(-1).astype(np.int8))
        previous.append(prev.reshape(-1))
    return np.concatenate(probabilities), np.concatenate(targets), np.concatenate(previous)


def choose_threshold(prob, target):
    valid = target >= 0
    pos = np.sort(prob[valid & (target == 1)])
    neg = np.sort(prob[valid & (target == 0)])
    if not len(pos) or not len(neg):
        raise ValueError('Validation must include both fire and non-fire labels')
    thresholds = np.arange(1, 100) / 100.
    tp = len(pos) - np.searchsorted(pos, thresholds, side='left')
    fp = len(neg) - np.searchsorted(neg, thresholds, side='left')
    fn = len(pos) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    curve = pd.DataFrame({'threshold': thresholds, 'f1': f1})
    return float(thresholds[np.argmax(f1)]), curve


def score_pixels(prob, target, threshold, mask):
    y = target[mask]
    p = prob[mask]
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))

    def ratio(a, b):
        return float(a / b) if b else None

    result = {'pixels': len(y), 'positive_pixels': int(np.sum(y == 1)),
              'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
              'f1_dice': ratio(2 * tp, 2 * tp + fp + fn),
              'iou': ratio(tp, tp + fp + fn),
              'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
              'pr_auc': None, 'average_precision': None}
    curve = None
    # PR area is undefined here if either class is absent. Constant predictions
    # can have misleading trapezoidal areas; also report AP and prevalence.
    if len(y) and np.any(y == 1) and np.any(y == 0):
        precision, recall, _ = precision_recall_curve(y, p)
        result['pr_auc'] = float(auc(recall, precision))
        result['average_precision'] = float(average_precision_score(y, p))
        # Thin only the exported plot, never the area calculation.
        stride = max(1, len(recall) // 2000)
        idx = np.unique(np.r_[np.arange(0, len(recall), stride), len(recall) - 1])
        curve = pd.DataFrame({'recall': recall[idx], 'precision': precision[idx]})
    result['prevalence'] = ratio(result['positive_pixels'], len(y))
    return result, curve


def evaluate(name, prob, target, prev, threshold):
    masks = {'all_observed_targets': target >= 0,
             'known_previous': (target >= 0) & (prev >= 0),
             'new_fire': (target >= 0) & (prev == 0)}
    reports = []
    for population, mask in masks.items():
        metrics, curve = score_pixels(prob, target, threshold, mask)
        reports.append({'model': name, 'population': population,
                        'threshold': threshold, **metrics})
        if curve is not None:
            curve.to_csv(OUTPUT_ROOT / f'{name}_{population}_pr.csv', index=False)
    return reports


def prediction_maps(name, model=None, keys=None, threshold=0.5):
    # Fixed first four test samples; no selection based on model performance.
    keep = tf.constant([float(k in (keys or FEATURES)) for k in FEATURES])
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    for row, (x, y) in enumerate(datasets['test'].take(4)):
        prev = x.numpy()[..., -1]
        target = y.numpy()[..., 0]
        prob = (prev == 1).astype(float) if model is None else (
            model((x * keep)[None], training=False).numpy()[0, ..., 0])
        predicted = np.where(target >= 0, prob >= threshold, -1)
        new = np.where((target >= 0) & (prev == 0), target, -1)
        for col, (values, title) in enumerate([
            (prev, 'Today'), (target, 'Tomorrow / target'),
            (prob, 'Fire probability'), (predicted, 'Prediction'),
            (new, 'New-fire target')]):
            ax = axes[row, col]
            im = ax.imshow(values, interpolation='nearest',
                           cmap='magma' if col == 2 else fire_cmap,
                           norm=None if col == 2 else fire_norm,
                           vmin=0 if col == 2 else None, vmax=1 if col == 2 else None)
            ax.set_title(title if row == 0 else f'Sample {row + 1}')
            ax.axis('off')
    fig.suptitle(f'{name}, validation-selected threshold={threshold:.2f}; gray=unknown/excluded')
    fig.savefig(OUTPUT_ROOT / f'{name}_prediction_maps.png', dpi=150)
    plt.show()
    plt.close(fig)


# Tests for metric counts, new-fire negatives, and a threshold perfect separation.
toy_p = np.array([0.9, 0.8, 0.1, 0.9])
toy_y = np.array([1, 0, 1, -1])
toy, _ = score_pixels(toy_p, toy_y, 0.5, toy_y >= 0)
assert (toy['tp'], toy['fp'], toy['fn']) == (1, 1, 1)
assert toy['f1_dice'] == 0.5
toy_threshold, _ = choose_threshold(np.array([0.2, 0.8]), np.array([0, 1]))
assert 0.2 < toy_threshold <= 0.8
print('Loss, architecture, and metric sanity checks passed.')

# %%
# Train all variants and lock their checkpoints/thresholds before touching test.
locked = {}
for name, keys in VARIANTS.items():
    print('\nTRAINING:', name, 'features:', keys, flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_unet()
    checkpoint = OUTPUT_ROOT / f'{name}.weights.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss',
                                          save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_ROOT / f'{name}_history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(feature_dataset('train', keys, training=True),
                        validation_data=feature_dataset('validation', keys),
                        epochs=EPOCHS, callbacks=callbacks, verbose=2)
    model.load_weights(checkpoint)
    prob, target, prev = collect_predictions('validation', model, keys)
    threshold, curve = choose_threshold(prob, target)
    curve.to_csv(OUTPUT_ROOT / f'{name}_validation_thresholds.csv', index=False)
    locked[name] = {'threshold': threshold, 'keys': keys,
                    'parameters': model.count_params(),
                    'epochs_run': len(history.history['loss']),
                    'best_epoch': int(np.argmin(history.history['val_loss']) + 1)}
    (OUTPUT_ROOT / 'locked_validation_choices.json').write_text(json.dumps(locked, indent=2))
    print(name, 'locked:', locked[name], flush=True)
    del model, prob, target, prev
    gc.collect()
print('All validation choices locked. Test evaluation can now run.')

# %%
results = []
prob, target, prev = collect_predictions('test')
results.extend(evaluate('persistence', prob, target, prev, 0.5))
prediction_maps('persistence')
del prob, target, prev
for name, choice in locked.items():
    tf.keras.backend.clear_session()
    model = build_unet()
    model.load_weights(OUTPUT_ROOT / f'{name}.weights.h5')
    prob, target, prev = collect_predictions('test', model, choice['keys'])
    results.extend(evaluate(name, prob, target, prev, choice['threshold']))
    prediction_maps(name, model, choice['keys'], choice['threshold'])
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / 'test_metrics.csv', index=False)
    del model, prob, target, prev
    gc.collect()
report = pd.DataFrame(results)
report.to_csv(OUTPUT_ROOT / 'test_metrics.csv', index=False)
print(report.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, population in zip(axes, ['all_observed_targets', 'new_fire']):
    for name in ['persistence'] + list(VARIANTS):
        path = OUTPUT_ROOT / f'{name}_{population}_pr.csv'
        if path.exists():
            curve = pd.read_csv(path)
            ax.plot(curve['recall'], curve['precision'], label=name)
    prevalence = report.loc[report['population'] == population, 'prevalence'].iloc[0]
    if pd.notna(prevalence):
        ax.axhline(prevalence, color='gray', linestyle='--', label='Positive prevalence')
    ax.set(title=population, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
fig.savefig(OUTPUT_ROOT / 'test_precision_recall.png', dpi=150)
plt.show()
print('COMPLETE. Artifacts:', OUTPUT_ROOT)
