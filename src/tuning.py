# %%
# Run after the first five cells of experiment.py; only train/validation are used.
TUNING_EPOCHS = 40
TUNING_PATIENCE = 8
TUNING_CANDIDATES = [
    {'name': 'shallow_longer', 'filters': 16, 'positive_weight': 1.},
    {'name': 'shallow_wider', 'filters': 32, 'positive_weight': 1.},
    {'name': 'shallow_wider_weighted', 'filters': 32, 'positive_weight': 6.},
]


def weighted_masked_bce_dice(y_true, y_pred, positive_weight=1.):
    valid = tf.cast(y_true >= 0, tf.float32)
    target = tf.where(y_true >= 0, y_true, 0.)
    prob = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
    axes = (1, 2, 3)
    count = tf.reduce_sum(valid, axis=axes)
    bce = -(positive_weight * target * tf.math.log(prob) +
            (1. - target) * tf.math.log(1. - prob))
    bce = tf.math.divide_no_nan(tf.reduce_sum(valid * bce, axis=axes), count)
    intersection = tf.reduce_sum(valid * target * prob, axis=axes)
    denominator = tf.reduce_sum(valid * (target + prob), axis=axes)
    dice_loss = 1. - (2. * intersection + 1.) / (denominator + 1.)
    observed = tf.cast(count > 0, tf.float32)
    return tf.math.divide_no_nan(
        tf.reduce_sum((0.5 * bce + 0.5 * dice_loss) * observed),
        tf.reduce_sum(observed))


def build_tuned_unet(filters, positive_weight):
    inputs = tf.keras.Input((64, 64, len(FEATURES)))
    skip = conv_block(inputs, filters, 'encoder')
    x = tf.keras.layers.MaxPooling2D()(skip)
    x = conv_block(x, 2 * filters, 'bottleneck')
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2)(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters, 'decoder')
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, output, name='tuned_shallow_unet')

    def loss(y_true, y_pred):
        return weighted_masked_bce_dice(y_true, y_pred, positive_weight)

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.),
                  loss=loss)
    return model


class ValidationDiceCheckpoint(tf.keras.callbacks.Callback):
    """Select a checkpoint and threshold jointly using pooled validation Dice."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.best_f1 = -1.
        self.best_epoch = None
        self.threshold = None
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        prob, target, _ = collect_predictions('validation', self.model, FEATURES)
        if not np.all(np.isfinite(prob)):
            raise FloatingPointError('Non-finite validation predictions; no checkpoint selected')
        threshold, curve = choose_threshold(prob, target)
        f1 = float(curve['f1'].max())
        logs['val_dice'] = f1
        self.records.append({'epoch': epoch + 1, 'val_dice': f1,
                             'threshold': threshold})
        pd.DataFrame(self.records).to_csv(
            OUTPUT_ROOT / f'{self.name}_validation_dice.csv', index=False)
        if f1 > self.best_f1:
            self.best_f1, self.best_epoch, self.threshold = f1, epoch + 1, threshold
            self.model.save_weights(OUTPUT_ROOT / f'{self.name}.weights.h5')
            curve.to_csv(OUTPUT_ROOT / f'{self.name}_validation_thresholds.csv', index=False)
        print(f'Validation Dice={f1:.4f}; threshold={threshold:.2f}; '
              f'best={self.best_f1:.4f} at epoch {self.best_epoch}', flush=True)


def validation_prediction_maps(name, model, threshold):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for row, (x, y) in enumerate(datasets['validation'].take(4)):
        target = y.numpy()[..., 0]
        prob = model(x[None], training=False).numpy()[0, ..., 0]
        predicted = np.where(target >= 0, prob >= threshold, -1)
        for col, (values, title) in enumerate([
            (x.numpy()[..., -1], 'Today'), (target, 'Tomorrow / target'),
            (prob, 'Fire probability'), (predicted, 'Prediction')]):
            ax = axes[row, col]
            image = ax.imshow(values, interpolation='nearest',
                cmap='magma' if col == 2 else fire_cmap,
                norm=None if col == 2 else fire_norm,
                vmin=0 if col == 2 else None, vmax=1 if col == 2 else None)
            ax.set_title(title if row == 0 else f'Validation sample {row + 1}')
            ax.axis('off')
            if col == 2:
                fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle(f'{name}; validation threshold={threshold:.2f}; gray=unknown')
    fig.savefig(OUTPUT_ROOT / f'{name}_validation_maps.png', dpi=150)
    plt.show()
    plt.close(fig)

# %%
# No test predictions or scores are computed by this tuning notebook.
metadata.update({
    'experiment': 'validation-only U-Net tuning',
    'epochs': TUNING_EPOCHS, 'patience': TUNING_PATIENCE,
    'candidates': TUNING_CANDIDATES,
    'checkpoint_selection': 'maximum pooled validation F1/Dice',
    'test_evaluation': 'not performed; original test has already been viewed',
    'lr_schedule': 'halve after 3 epochs without improving validation loss; min 1e-5',
})
(OUTPUT_ROOT / 'config.json').write_text(json.dumps(metadata, indent=2))
choices, validation_results = {}, []
for candidate in TUNING_CANDIDATES:
    name = candidate['name']
    print('\nVALIDATION TUNING:', candidate, flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_tuned_unet(candidate['filters'], candidate['positive_weight'])
    selector = ValidationDiceCheckpoint(name)
    callbacks = [
        selector,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3,
            factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='val_dice', mode='max',
            patience=TUNING_PATIENCE, restore_best_weights=False),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_ROOT / f'{name}_history.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(feature_dataset('train', FEATURES, training=True),
        validation_data=feature_dataset('validation', FEATURES),
        epochs=TUNING_EPOCHS, callbacks=callbacks, verbose=2)
    model.load_weights(OUTPUT_ROOT / f'{name}.weights.h5')
    choices[name] = {**candidate, 'threshold': selector.threshold,
        'best_epoch': selector.best_epoch, 'validation_dice': selector.best_f1,
        'epochs_run': len(history.history['loss']), 'parameters': model.count_params()}
    (OUTPUT_ROOT / 'locked_validation_choices.json').write_text(json.dumps(choices, indent=2))
    prob, target, previous = collect_predictions('validation', model, FEATURES)
    validation_results.extend(evaluate(name, prob, target, previous, selector.threshold))
    pd.DataFrame(validation_results).to_csv(OUTPUT_ROOT / 'validation_metrics.csv', index=False)
    validation_prediction_maps(name, model, selector.threshold)
    del model, prob, target, previous
    gc.collect()
winner = max(choices, key=lambda name: choices[name]['validation_dice'])
(OUTPUT_ROOT / 'selected_model.json').write_text(json.dumps(choices[winner], indent=2))
print(pd.DataFrame(validation_results).to_string(index=False))
print('Selected by validation Dice only:', winner)
print('VALIDATION TUNING COMPLETE. No test evaluation performed. Artifacts:', OUTPUT_ROOT)
