import sqlite3
from settings import DEV_DB_PATH


class DataModel:
    def __init__(self):
        self.con = sqlite3.connect(DEV_DB_PATH)

    def test(self):
        cur = self.con.cursor()
        for i, row in enumerate(
                cur.execute('SELECT language, snippet FROM snippets;')):
            if i > 100:
                break
            print(row)

    def __exit__(self):
        self.con.close()


if __name__ == '__main__':
    DataModel().test()