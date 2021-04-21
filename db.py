import sqlite3
import tensorflow as tf
from settings import DEV_DB_PATH


def load(dev=True):
    if dev:
        return tf.data.experimental.SqlDataset(
            "sqlite",
            DEV_DB_PATH,
            "select language, snippet FROM snippets;",
            (tf.string, tf.string),
        )
    else:
        raise Exception("Full database not stored")


if __name__ == "__main__":
    db = load()
    for i, elt in enumerate(db):
        if i > 1000:
            break
        print(elt)