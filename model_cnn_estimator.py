import tensorflow as tf


MODE = {
    'pred': tf.estimator.ModeKeys.PREDICT,
    'eval': tf.estimator.ModeKeys.EVAL,
    'train': tf.estimator.ModeKeys.TRAIN

}


def model(input_layer):

    layer1 = tf.layers.Conv2D(inputs=input_layer,
                              filters=32,
                              kernel_size=[2,2],
                              strides=1,
                              activation=tf.nn.softmax,
                              padding="same")

    pooling_layer1 = tf.layers.max_pooling2d(layer1, pool_size=[2,2], strides=2)

    layer2 = tf.layers.Conv2D(inputs=pooling_layer1,
                              filters=12,
                              kernel_size=[2,2],
                              strides=1,
                              padding='same',
                              activation='tf.nn.relu')

    pooling_layer2 = tf.layers.max_pooling2d(layer2, pool_size=[2,2], strides=1)

    dense_layer = tf.layers.dense(inputs=tf.reshape(pooling_layer2, [-1, 1024]),
                                  units=1024,
                                  activation=tf.nn.relu)

    return dense_layer


def get_logits(layer, units):

    return tf.layers.logits(layer, units=units)


def get_predictions(logits, input=1):

    return tf.argmax(logits, input)


def check_mode_pred(mode, logits, predicted_classes):

    if mode == MODE['pred']:
        predictions = {
            'probabilties': tf.nn.softmax(logits),
            'logits': logits,
            'prediction': predicted_classes[:, tf.newaxis]

        }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def check_mode_train(mode, units):

    if mode == MODE['train']:
        return tf.estimator.EstimatorSpec(mode=mode, units=units)


def check_mode_eval(mode, loss, metrics):
    if mode == MODE['eval']:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def get_loss(labels, logits):

    return tf.losses.softmax_cross_entropy(labels=labels, logits=logits)


def run_model(features, labels, mode, params, units):

    input_layer = tf.feature_column.input_layer(features, params['feature_column'])

    _model = model(input_layer)

    logits = get_logits(_model, units=units)

    prediction = get_predictions(logits)
    check_mode_pred(mode, logits, predicted_classes=prediction)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=prediction,
                                   name="acc_op")

    metrics = {'accuracy': accuracy}

    tf.summary.scalar('accuracy', accuracy[1])

    check_mode_eval(mode, get_loss(labels, logits), metrics=metrics)

    check_mode_train(mode)


    return