import pandas as pd
import tensorflow as tf
import os


features_column = [
    'feature_one',
    'feature_two',
    'feature_three',
    'feature_four',
    'feature_five',
    'feature_six',
    'feature_seven',
    'feature_eight',
    'feature_nine',
    'feature_ten',
    'feature_eleven',
    'feature_twelve',
    'feature_thirteen',
    'label',
    ]

def getDataset():
   # url to file
   data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
   #data_url = "http://download.tensorflow.org/data/iris_training.csv"
   # return file path of the data
   return tf.keras.utils.get_file(os.path.basename(data_url), origin=data_url)

def data_url():
    return "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def prepare_data(label='label'):

    _data = pd.read_csv(getDataset(), names=features_column)

    features, labels = _data, _data.pop(label)


    return features, labels


def make_data_tensors(features, label, batch_size):
    # tf.data.Dataset.from_tensor_slices
    _data = tf.data.Dataset.from_tensor_slices((dict(features), label))

    _data = _data.shuffle(1000).repeat().batch(batch_size=batch_size)

    return _data


def get_array_values(features, labels):

    return features.get_values(), labels.get_values()




# features, labels = prepare_data()

# print(features.get_values())


# print(data_url())
# print(prepare_data())
# print(make_data_tensors(features, labels, 32))