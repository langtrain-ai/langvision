class Callback:
    def on_train_begin(self, trainer):
        pass
    def on_epoch_begin(self, trainer, epoch):
        pass
    def on_batch_end(self, trainer, batch, logs=None):
        pass
    def on_epoch_end(self, trainer, epoch, logs=None):
        pass
    def on_train_end(self, trainer):
        pass 