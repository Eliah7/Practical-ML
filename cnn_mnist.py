from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#Our application will be added here
def cnn_model_fn(features,labels,mode):
    """Model function for my CNN"""
    # Input Layer
    input_layer = tf.reshape(feature["x"],[-1,28,28,1],name='input_layer')

    # Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    

"""if __name__ == "__main__":
    tf.app.run()"""
