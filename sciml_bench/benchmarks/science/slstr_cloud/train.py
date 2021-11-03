import tensorflow as tf
import horovod.tensorflow as hvd

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

def train_model(train_dataset, test_dataset, model, args:dict,  \
                            params_in: RuntimeIn, params_out:RuntimeOut) -> None:
    
    """
    Train a U-Net style model
    """
    console = params_out.log.console
    device = params_out.log.device

    learning_rate = args['learning_rate']
    epochs = args['epochs']
    wbce = args['wbce']
    clip_offset = args['clip_offset']
    
 
    # Setup the loss functions and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bce = weighted_cross_entropy(wbce)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    log = params_out.log

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
            params_out.system.stamp_event(f'epoch {epoch}: train')
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
            params_out.system.stamp_event(f'epoch {epoch}: validate')
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
            model_file = params_in.output_dir / 'model.h5'
            model.save(model_file)

            history.append(dict(train_accuracy=train_accuracy, epoch=epoch, train_loss=train_loss,
                                test_accuracy=test_accuracy, test_loss=test_loss))

    return history 

