import time
import tensorflow as tf

class LogEpochCallback(tf.keras.callbacks.Callback):
    """
    Callback to log epoch 
    """

    def __init__(self, params_out):
        super().__init__()
        self._start_time = time.time()
        self._params_out = params_out

    def on_epoch_begin(self, epoch, logs=None):
        # stamp epoch in system monitor
        self._start_time = time.time()
        self._params_out.system.stamp_event(f'epoch {epoch}')

    def on_epoch_end(self, epoch, logs=None):
        msg = f'Epoch {epoch:2d}: '
        for key, val in logs.items():
            msg += f'{key}={val:f} '
        msg += f'elapsed={time.time() - self._start_time:f} sec'
        self._params_out.log.message(msg)
