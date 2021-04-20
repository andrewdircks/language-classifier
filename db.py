import sqlite3
import tensorflow as tf
from settings import DEV_DB_PATH


class DataModel:
    def __init__(self):
        self.con = sqlite3.connect(DEV_DB_PATH)

    def load(self):
        return tf.data.experimental.SqlDataset(
            "sqlite",
            DEV_DB_PATH,
            "select language, snippet FROM snippets;",
            (tf.string, tf.string),
        )

    def test_tf(self):
        dataset = self.load()
        for i, elt in enumerate(dataset):
            if i > 100:
                break
            print(elt)

    def __exit__(self):
        self.con.close()


if __name__ == "__main__":
    DataModel().test_tf()