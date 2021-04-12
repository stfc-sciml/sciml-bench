from os.path import split

import yaml
import tensorflow as tf
import horovod.tensorflow as hvd
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data_loader import SLSTRDataLoader
from .model import unet
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut


def weighted_cross_entropy(beta):
    """
    Weighted Binary Cross Entropy implementation

    :param beta: beta weight to adjust relative importance of +/- label
    :return: weighted BCE loss
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=y_pred, labels=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


def train_model(daatset_dir: Path, args: dict, smlb_in: RuntimeIn, smlb_out: RuntimeOut) -> None:
    """
    Train a U-Net style model
    :param daatset_dir: path to the data files
    :param args: dictionary of user/environment arguments
    :param smlb_in: RuntimeIn instance for logging
    :param smlb_out: RuntimeOut instance for logging
    """
    console = smlb_out.log.console
    device = smlb_out.log.device

    learning_rate = args['learning_rate']
    epochs = args['epochs']
    batch_size = args['batch_size']
    wbce = args['wbce']
    clip_offset = args['clip_offset']

    # Pin the number of GPUs to the local rank for Horovod
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()], 'GPU')

    if hvd.rank() == 0:
        console.message(f"Num GPUS: {len(gpus)}")
        console.message(f"Num ranks: {hvd.size()}")

    # Get the data loader
    data_paths = list(Path(daatset_dir).glob('**/S3A*.hdf'))
    train_paths, test_paths = train_test_split(data_paths, train_size=args['train_split'], random_state=42)

    train_data_loader = SLSTRDataLoader(train_paths, batch_size=batch_size, no_cache=args['no_cache'])
    train_dataset = train_data_loader.to_dataset()

    test_data_loader = SLSTRDataLoader(test_paths, batch_size=batch_size, no_cache=args['no_cache'])
    test_dataset = test_data_loader.to_dataset()

    model = unet(train_data_loader.input_size)

    # Setup the loss functions and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bce = weighted_cross_entropy(wbce)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    log = smlb_out.log

    @tf.function
    def train_step(images, masks, first_batch=False):
        with tf.GradientTape() as tape:
            predicted = model(images)
            predicted = predicted[:, clip_offset:-
                                  clip_offset, clip_offset:-clip_offset]
            masks = masks[:, clip_offset:-
                          clip_offset, clip_offset:-clip_offset]
            loss = bce(masks, predicted)
            train_loss_metric.update_state(loss)

        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(
            loss, model.trainable_variables)

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return predicted, masks

    @tf.function
    def test_step(images, masks, first_batch=False):
        predicted = model(images)
        predicted = predicted[:, clip_offset:-
        clip_offset, clip_offset:-clip_offset]
        masks = masks[:, clip_offset:-
        clip_offset, clip_offset:-clip_offset]
        loss = bce(masks, predicted)
        test_loss_metric.update_state(loss)
        return predicted, masks

    history = []
    for epoch in range(epochs):
        # Clear epoch metrics
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()

        test_loss_metric.reset_states()
        test_acc_metric.reset_states()

        with log.subproc(f'Epoch {epoch}'):
            device.begin(f"Epoch {epoch}")

            # Train model
            device.begin("Training")
            smlb_out.system.stamp_event(f'epoch {epoch}: train')
            for i, (images, masks) in enumerate(train_dataset):
                predicted, msk = train_step(images, masks, i == 0)
                train_acc_metric.update_state(msk, predicted)
                message = f'Batch: {i}, Train Loss: {train_loss_metric.result().numpy(): .5f}'
                log.message(message)
                if hvd.rank() == 0:
                    device.message(message)

            device.ended("Training")
            device.begin("Testing")

            # Test model
            smlb_out.system.stamp_event(f'epoch {epoch}: validate')
            for i, (images, masks) in enumerate(test_dataset):
                predicted, msk = test_step(images, masks, i == 0)
                test_acc_metric.update_state(msk, predicted)
                message = f'Batch: {i}, Test Loss: {test_loss_metric.result().numpy(): .5f}'
                log.message(message)
                if hvd.rank() == 0:
                    message = f'Batch: {i}, Test Loss: {test_loss_metric.result().numpy(): .5f}'
                    device.message(message)

            device.ended('Testing')
            device.ended(f"Epoch {epoch}")

        # Log Epoch Results
        if hvd.rank() == 0:
            # Print metrics
            train_loss = train_loss_metric.result().numpy()
            train_accuracy = train_acc_metric.result().numpy()

            test_loss = test_loss_metric.result().numpy()
            test_accuracy = test_acc_metric.result().numpy()

            message = f'Epoch {epoch}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, ' \
                      f'Train Acc: {train_accuracy: .2f}, Test Acc: {test_accuracy: .2f}'
            console.message(message)

            # Save model
            model_file = smlb_in.output_dir / 'model.h5'
            model.save(model_file)

            history.append(dict(train_accuracy=train_accuracy, epoch=epoch, train_loss=train_loss,
                                test_accuracy=test_accuracy, test_loss=test_loss))

    history_file = smlb_in.output_dir / 'training_history.yml'
    with history_file.open('w') as handle:
        yaml.dump(history, handle)

