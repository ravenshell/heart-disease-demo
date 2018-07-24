from __future__ import division, absolute_import, print_function

import matplotlib as plt
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from heart_disease import data as dd

tf.enable_eager_execution()


# TODO: create a class for training individual models


def create_model(first_layer=10, second_layer=10, input_layer =13 ):
   model = tf.keras.Sequential([
       tf.keras.layers.Dense(10, input_shape=(input_layer,)),
       tf.keras.layers.Dense(700, activation="tanh"),
       tf.keras.layers.Dense(1022, activation='sigmoid'),
       tf.keras.layers.Dense(2, activation='sigmoid'),
       tf.keras.layers.Dense(5)

       # 54 % at 0.5 learning rate
       # tf.keras.layers.Dense(10, input_shape=(input,)),
       # tf.keras.layers.Dense(5, activation="sigmoid"),
       # # tf.keras.layers.Dense(20, activation='sigmoid'),
       # tf.keras.layers.Dense(1022, activation='sigmoid'),
       # tf.keras.layers.Dense(5)

   ])

   return model


def new_train(features, labels, epochs = 3):

    model = create_model()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(features, labels, epochs)

    return model


def eval(model, eval_x, eval_y):

    return model.evaluate(eval_x, eval_y, batch_size=30)




# Test Keras

# features, labels = dd.prepare_data()
# _model = new_train(features.get_values(), labels.get_values(), epochs=5)
#
# eval(_model, features, labels)
#









def loss(model, x, y):
   # do prediction
   y_cap = model(x)
   # get the loss
   return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_cap)


def gradnt(model, _input, _output):
   # tfe.GradientTape is used to record "operations"
   # that "compute the gradients" used to optimize our model
   with tfe.GradientTape() as tape:
       loss_value = loss(model, _input, _output)
   return tape.gradient(loss_value, model.variables)


def getGradientDescentOptimizer(learning_rate=1.2):
   return tf.train.GradientDescentOptimizer(learning_rate)


def train(data_file, epoch=2000):
   training_losses = []
   accuracy_results = []

   data = dd.prepare_data(data_file)
   # print(data)

   # get label and feature  data from
   # prepare_data() function using the tfe.Iterator().next()

   # Note: the line retrieves just the first
   # line from the dataset. use for loop to loop
   # through the entrie dataset

   # labels, features = tfe.Iterator(data).next()




   # train for over 400 epochs
   for _epoch in range(epoch):
       mean = tfe.metrics.Mean()
       accuracy = tfe.metrics.Accuracy()

       for x, y in tfe.Iterator(data):
           model = create_model(first_layer=10, second_layer=5)

           # optimize model
           grad = gradnt(model, x, y)
           getGradientDescentOptimizer().apply_gradients(zip(grad, model.variables), global_step=tf.train.get_or_create_global_step())

           # add current batch loss
           mean(loss(model, x, y))

           # do accuracy test:
           #       compare predicted to actual value
           accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

       if _epoch % 5 == 0:
           print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(_epoch,
                                                                       mean.result(),
                                                                       accuracy.result()))


train(dd.getDataset())