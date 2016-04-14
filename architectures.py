import tensorflow as tf

from layers import Dense, Conv2D, Flatten, Conv2DBatchNorm, AvgPool, Dropout, Activation

def vgg_bn():
    return [
        Conv2D([3, 3], 32, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(32),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 32, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(32),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 64, [1, 2, 2, 1]),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 64, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(64),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 128, [1, 2, 2, 1]),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),

        Conv2D([3, 3], 128, [1, 1, 1, 1], padding='SAME'),
        Conv2DBatchNorm(128),
        Activation(tf.nn.relu),

        Flatten(),

        Dense(128),
        Activation(tf.sigmoid),

        Dropout(0.5),

        Dense(10),
        Activation(tf.nn.softmax),
    ]
