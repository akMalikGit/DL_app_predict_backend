import tensorflow as tf
import matplotlib.pyplot as plt


def draw_history_graph(history, out_dtype):
    # summarize history for accuracy
    if out_dtype == tf.float32:
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('model mean_absolute_error')
        plt.ylabel('mean_absolute_error')
    else:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()



class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, val_loss_threshold, val_mae_threshold=None, val_acc_threshold=None, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.val_loss_threshold = val_loss_threshold
        self.val_mae_threshold = val_mae_threshold
        self.val_acc_threshold = val_acc_threshold
        self.patience = patience
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')


        if self.val_mae_threshold:
            val_acc = logs.get('val_mean_absolute_error')
            if val_loss < self.val_loss_threshold and val_acc < self.val_mae_threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
            else:
                self.wait = 0
                self.best_weights = self.model.get_weights()
        else:
            val_acc = logs.get('val_accuracy')

            if val_loss < self.val_loss_threshold and val_acc > self.val_acc_threshold:
                self.wait += 1
                # print('\n\tepoch=', epoch)
                # print('\twait=', self.wait)
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
            else:
                # print('\n\twait reset=', self.wait)
                self.wait = 0
                self.best_weights = self.model.get_weights()


def custom_callbacks(out_dtype):
    if out_dtype == tf.float32:
        return CustomEarlyStopping(val_loss_threshold=0.1, val_mae_threshold=0.1, val_acc_threshold=None, patience=20)
    return CustomEarlyStopping(val_loss_threshold=0.1, val_mae_threshold=None, val_acc_threshold=0.95, patience=20)
