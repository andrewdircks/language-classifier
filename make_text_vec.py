import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from db import load, split_dataset, process_label
from settings import PERCENT_TEST, MAX_FEATURES, SNIPPET_LENGTH, BATCH_SIZE, n_langs

if __name__ == '__main__':
    data = tf.data.experimental.SqlDataset(
        "sqlite",
        "snippets-dev/snippets-dev.db",
        "select snippet, language FROM snippets;",
        (tf.string, tf.int8),
    )

    train, test = split_dataset(data, PERCENT_TEST)

    # for label, snippet in train.take(1):
    #   print('label: ', label.numpy())
    #   print('snippet: ', snippet.numpy())
    
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test = test.batch(BATCH_SIZE)


    # for labels, example in train.take(1):
    #   print('texts: ', example.numpy()[:3])
    #   print()
    #   _label = [process_label(l) for l in labels[:3]]
    #   print('labels: ', _label)


    #  build vocab and encoder
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train.map(lambda snippet, label: snippet))


    # vocab = np.array(encoder.get_vocabulary())
    # print(vocab[:20])

    vocab_len = len(np.array(encoder.get_vocabulary()))

    # model1 = tf.keras.Sequential([
    #           encoder,
    #           tf.keras.layers.Embedding(
    #               input_dim=len(encoder.get_vocabulary()),
    #               output_dim=64,
    #               # Use masking to handle the variable sequence lengths
    #               mask_zero=True),
    #           # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #           tf.keras.layers.Flatten(),
    #           tf.keras.layers.Dense(64, activation='relu'),
    #           tf.keras.layers.Dense(n_langs)
    #       ])


    # model1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #           optimizer=tf.keras.optimizers.Adam(1e-4),
    #           metrics=['accuracy'])

    model = tf.keras.Sequential([
              encoder,
              tf.keras.layers.Embedding(
                  input_dim=len(encoder.get_vocabulary()),
                  output_dim=64,
                  # Use masking to handle the variable sequence lengths
                  mask_zero=False),
              # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dense(n_langs)
          ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['accuracy'])


    # train
    history = model.fit(train, epochs=1,
                    validation_data=test,
                    validation_steps=30)
    
    history = model.fit(train.map(lambda snippet, label: (encoder(snippet), label)))


    # test
    test_loss, test_acc = model.evaluate(test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    
    model.save("test.h5")

