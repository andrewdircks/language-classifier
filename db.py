import sqlite3
import numpy as np
import tensorflow as tf
from settings import (
    DEV_DB_PATH,
    FULL_DB_PATH,
    langs_map,
    chars_map,
    n_chars,
    snippet_len,
)


def get_label(entry):
    """
    Return this data entry's label as a byte variable.
    """
    return tf.Variable(langs_map[entry], dtype="int8")


def get_snippet(entry):
    """
    Return this data entry's code snippet as a byte vector with.
    """
    # read individual characters of tf string
    entry = tf.strings.unicode_split(entry, "UTF-8").numpy()

    freqs = np.zeros([n_chars], dtype="int8")
    for i, c in enumerate(entry):
        if i > snippet_len:
            break
        index = chars_map[c.decode("UTF-8").lower()]
        freqs[index] += 1

    print(freqs)
    return tf.Variable(freqs, dtype="int8")


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


if __name__ == "__main__":
    data = load()
    data = data.shuffle(1000)
    for i, elt in enumerate(data):
        if i > 1000:
            break
        v = get_snippet(elt[1])
        print(v)
    # get_snippet("")
