import tensorflow as tf
from utils import TrainerClass


class TrainModel:

    def __init__(self, model, train_ds, test_ds):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds

    def masked_loss(labels, preds):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

        mask = (labels != 0) & (loss < 1e8)
        mask = tf.cast(mask, loss.dtype)

        loss = loss * mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss

    def masked_acc(labels, preds):
        mask = tf.cast(labels != 0, tf.float32)
        preds = tf.argmax(preds, axis=-1)
        labels = tf.cast(labels, tf.int64)
        match = tf.cast(preds == labels, mask.dtype)
        acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
        return acc

    def model_train(self):
        callbacks = [
            GenerateText(),
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True)]

        g = GenerateText()
        g.model = self.model
        g.on_epoch_end(0)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss=self.masked_loss,
                           metrics=[self.masked_acc])

        self.model.fit(
            self.train_ds.repeat(),
            steps_per_epoch=100,
            validation_data=self.test_ds.repeat(),
            validation_steps=20,
            epochs=2,
            callbacks=callbacks)


class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
        self.image = TrainerClass().load_image(image_path)

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()
