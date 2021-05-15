import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input, Conv1D, MaxPooling1D
from db import _map
from settings import BATCH_SIZE, NUM_EPOCHS, n_chars, n_langs


# if __name__ == '__main__':
#   mapped_data = _map()
#   print(mapped_data)

#   inputs = keras.Input(shape=(n_chars,), name="chars")
#   x1 = Dense(64, activation="relu", name="dense_1")(inputs)
#   x2 = Dense(64, activation="relu", name="dense_2")(x1)
#   outputs = Dense(n_langs, name="predictions")(x2)
#   model = keras.Model(inputs=inputs, outputs=outputs)

#   model.add_loss(tf.reduce_sum(x1) * 0.1)
#   model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")

#   # model = Sequential()
#   # model.add(Dense(n_chars, input_dim=n_chars))
#   # model.add(Dense(n_langs, activation='softmax'))
#   model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#   # print(tf.shape(mapped_data))
#   model.fit(mapped_data, y=None, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=True)
#   model.save("test.h5")

if __name__ == '__main__':
    training_data = _map()
    print(training_data)

    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(n_langs)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
    model.fit(training_data, epochs=NUM_EPOCHS, verbose=True)