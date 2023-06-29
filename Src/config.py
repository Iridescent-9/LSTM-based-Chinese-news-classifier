from datetime import datetime
import os


class Config(object):
    def __init__(self):
        timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.config_dict = {
            "data_path": {
                "vocab_path": "../Data/cnews.vocab.txt",
                "trainingSet_path": "../Data/cnews.train.txt",
                "valSet_path": "../Data/cnews.val.txt",
                "testingSet_path": "../Data/cnews.test.txt"
            },
            "LSTM": {
                "seq_length": 600,
                "num_classes": 10,
                "vocab_size": 5000,
                "batch_size": 64,
                "log_dir": os.getcwd() + "\\Logs\\" + timestamp
            },
            "result": {
                "LSTM_model_path": "../Result/LSTM_model.h5"
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]
