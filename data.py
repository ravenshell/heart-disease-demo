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

#  -------------------------------Using tf.data ---------------------------------------------------



# data preparation

# these should be added to constructor
input = 13
#field_type

def prepare_data(data_file):
   # instantiate TextLineDataset(data_file) that will
   # be used to retrive content of data_file
   training_data = tf.data.TextLineDataset(data_file)
   #training_data = training_data.skip(1)

   # apply parse_csv_data() function to the data
   training_data = training_data.map(parse_csv_data)

   # random shuffle
   training_data = training_data.shuffle(2000)
   # batch size must greater than 30
   # to attain the normal distribution
   training_data = training_data.batch(50)


   return training_data


# def getDataset():
#    # url to file
#    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#    #data_url = "http://download.tensorflow.org/data/iris_training.csv"
#    # return file path of the data
#    return tf.keras.utils.get_file(os.path.basename(data_url), origin=data_url)
#

def parse_csv_data(csv_current_line):
   # set the field type of each entry from our csv dataset
   #fields_type = [[0.], [0.], [0.], [0.], [0]]
   fields_type = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0]]

   # decode csv using tf.decode_csv() function
   data = tf.decode_csv(csv_current_line, fields_type)

   # seperate fetures from ouput

   # strip the first 13 features into a single tensor
   # print(data[:-1])
   features = tf.reshape(data[:-1], shape=(input,))
   # print(features)

   # strip the last feature as our label
   labels = tf.reshape(data[-1], shape=())

   return features, labels
