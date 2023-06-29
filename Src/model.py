import tensorflow as tf
from config import Config
from preprocess import preprocesser
import os
from sklearn import metrics
import numpy as np


class LSTM(object):

    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()

    def model(self):
        seq_length = self.config.get("LSTM", "seq_length")
        num_classes = self.config.get("LSTM", "num_classes")
        vocab_size = self.config.get("LSTM", "vocab_size")

        model_input = tf.keras.layers.Input(seq_length)
        embedding = tf.keras.layers.Embedding(vocab_size + 1, 256, input_length=seq_length)(model_input)
        LSTM = tf.keras.layers.LSTM(256)(embedding)
        FC1 = tf.keras.layers.Dense(256, activation="relu")(LSTM)
        droped = tf.keras.layers.Dropout(0.5)(FC1)
        FC2 = tf.keras.layers.Dense(num_classes, activation="softmax")(droped)

        model = tf.keras.models.Model(inputs=model_input, outputs=FC2)

        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=["accuracy"])
        model.summary()
        # img = keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
        return model

    def train(self, epochs):
        training_set_path = self.config.get("data_path", "trainingSet_path")
        val_set_path = self.config.get("data_path", "valSet_path")
        seq_length = self.config.get("LSTM", "seq_length")
        model_save_path = self.config.get("result", "LSTM_model_path")
        log = self.config.get("LSTM", "log_dir")
        batch_size = self.config.get("LSTM", "batch_size")
        model = self.model()

        x_train, y_train = self.pre.word2idx(training_set_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(val_set_path, max_length=seq_length)
        tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_val, y_val),
                  epochs=epochs,
                  callbacks=[tf_callback])
        print("-----model save-----")
        model.save(model_save_path, overwrite=True)

    def test(self):
        model_save_path = self.config.get("result", "LSTM_model_path")
        testingSet_path = self.config.get("data_path", "testingSet_path")
        seq_length = self.config.get("LSTM", "seq_length")

        if os.path.exists(model_save_path):
            model = tf.keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        x_test, y_test = self.pre.word2idx(testingSet_path, max_length=seq_length)
        pre_test = model.predict(x_test)
        print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))


if __name__ == '__main__':
    LSTMTest = LSTM()
    LSTMTest.train(10)
    LSTMTest.test()
