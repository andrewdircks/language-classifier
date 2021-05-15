import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from settings import (
    DEV_DB_PATH,
    FULL_DB_PATH,
    langs_map,
    chars_map,
    n_chars,
    snippet_len,
    BATCH_SIZE
)


def process_label(label):
    """
    Return this data entry's label as a byte variable. Assume `entry` is an 
    element of `settings.langs`.
    """
    label = label.numpy().decode('UTF-8')
    return tf.convert_to_tensor(langs_map[label], dtype=tf.int8)


def _to_byte_vector(char_dict):
    vec = np.zeros([n_chars], dtype='int8')
    for char in char_dict:
        try:
            vec[chars_map[char]] += char_dict[char]
        except:
            pass
    return tf.convert_to_tensor(vec, dtype=tf.int8)


def process_snippet(tok, snippet):
    """
    Return this data entry's code snippet as a byte vector with...
    """
    tok = Tokenizer(char_level=True)
    tok.fit_on_texts([snippet.numpy().decode('UTF-8')])
    return _to_byte_vector(tok.word_index)


def process(tok, label, snippet):
    return process_snippet(tok, snippet), process_label(label)

def process_pyfn(label, snippet):
    tok = Tokenizer(char_level=True)
    x, y = tf.py_function(lambda label, snippet: process(tok, label, snippet), inp=[label, snippet], Tout=[tf.int8, tf.int8])
    x.set_shape((n_chars,))
    y.set_shape(())
    return x, y

def load(dev=True):
    if dev:
        db_path = DEV_DB_PATH
    elif FULL_DB_PATH:
        db_path = FULL_DB_PATH
    else:
        raise Exception("Full database not stored")

    return tf.data.experimental.SqlDataset(
        "sqlite",
        db_path,
        "select language, snippet FROM snippets;",
        (tf.string, tf.string),
    )

def _map():
    data = load()
    data = data.shuffle(BATCH_SIZE)
    data = data.map(process_pyfn).batch(BATCH_SIZE)
    return data

if __name__ == "__main__":
    data = _map()
    for i, d in enumerate(data):
        if i > 0:
            break
        print(d)