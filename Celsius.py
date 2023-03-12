import tensorflow as tf
import logging
import numpy as np


#records errors
logging.getLogger("tensorflow").setLevel(logging.ERROR)

celsius_input = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farenheit_output = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_input):
    print("{} degress celsius = {} degrees farenheit".format(c, farenheit_output[i]))

#layer1 of a dense network
layer1 = tf.keras.layers.Dense(units=1, input_shape=[1])

#model created from 1 layer
model = tf.keras.Sequential(layer1)

#model compiled
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

#train model, 1 epoch is 1 example, 500 epcohs times 7 values means it was trained with 3500 examples
history = model.fit(celsius_input, farenheit_output, epochs=500, verbose=False)
print("model finished training")


#predict result for 100C to f
print("100C to F is 212, Model predicted {}".format(model.predict([100.0])))


model.compile