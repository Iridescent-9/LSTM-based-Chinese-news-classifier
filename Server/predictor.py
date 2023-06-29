from os import path

from numpy import argmax
from tensorflow import keras
from Src.config import Config
from Src.preprocess import preprocesser


class Predictor:
    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()
        self.categories = ["财经", "房产", "家居", "教育", "科技", "时尚", "时政", "体育", "游戏", "娱乐"]
        # categories = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座",
        #               "游戏", "娱乐"]
        self.model_path = self.config.get("result", "LSTM_model_path")
        if path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)

    def predict(self, articles):
        articles_text = "".join(articles)
        seq_length = self.config.get("LSTM", "seq_length")
        x_test = self.pre.word2idx_for_sample(articles_text, max_length=seq_length)
        pre_test = self.model.predict(x_test)
        predicted_labels = [self.categories[argmax(prediction)] for prediction in pre_test]
        print(predicted_labels)
        return predicted_labels
