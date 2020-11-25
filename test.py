# 2018.3.1
# Learn structure of Tensorflow
# episode 1
# 
# 

import tensorflow as tf
import numpy as np

train_X = np.linspace(-1, 1, 100)
train_X = np.expand_dims(train_X, axis=-1)

train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10


# First create a  model with one unit of dense and one bias
input = tf.keras.layers.Input(shape=(1,))
w = tf.keras.layers.Dense(1)(input)   # use_bias is True by default
model = tf.keras.Model(inputs=input, outputs=w)

opt=tf.keras.optimizers.SGD(0.1)
mse=tf.keras.losses.MeanSquaredError()

for i in range(20):
    print('Epoch: ', i)
    with tf.GradientTape() as grad_tape:
        logits = model(train_X, training=True)
        model_loss = mse(train_Y, logits)
        print('Loss =', model_loss.numpy())

    gradients = grad_tape.gradient(model_loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
