import tensorflow as tf


class AODNet(tf.keras.Model):

    def __init__(self, stddev: float = 0.02, weight_decay: float = 1e-4):
        super(AODNet, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu,
            use_bias=True, kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu,
            use_bias=True, kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        )
        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu,
            use_bias=True, kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        )
        self.conv_layer_4 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu,
            use_bias=True, kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        )
        self.conv_layer_5 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
            use_bias=True, kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
        )
        self.relu = tf.keras.layers.ReLU(max_value=1.0)

    def call(self, inputs, *args, **kwargs):
        conv_1 = self.conv_layer_1(inputs)
        conv_2 = self.conv_layer_2(conv_1)
        concat_1 = tf.concat([conv_1, conv_2], axis=-1)
        conv_3 = self.conv_layer_3(concat_1)
        concat_2 = tf.concat([conv_2, conv_3], axis=-1)
        conv_4 = self.conv_layer_4(concat_2)
        concat_3 = tf.concat([conv_1, conv_2, conv_3, conv_4], axis=-1)
        k = self.conv_layer_5(concat_3)
        j = tf.math.multiply(k, inputs) - k + 1.0
        output = self.relu(j)
        return output
