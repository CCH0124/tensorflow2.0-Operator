import tensorflow as tf
class MyStopTrainCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('loss') < 0.3):
            print("\nLoss is low so cancelling training!!")
            self.model.stop_training = True