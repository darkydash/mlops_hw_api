from contextlib import contextmanager

import psycopg2
import os


def log_to_db(error_code, error_message):
    if os.getenv('TEST'):
        return

    db_util = DbUtil()
    db_util.log(error_code, error_message)


class DbUtil:
    def __init__(
            self,
            username='postgres',
            password='postgres',
            db='ml_service',
            host='db'
    ):
        self.dbname = db
        self.user = username
        self.password = password
        self.host = host

    def log(self, error_code, message):
        with self.get_cursor() as c:
            c.execute(
                'INSERT INTO error_logs (error_code, error_message) VALUES (%s, %s)',
                (error_code, message)
            )

    @contextmanager
    def get_cursor(self):
        conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host
        )

        cursor = conn.cursor()

        try:
            yield cursor
            conn.commit()
        finally:
            cursor.close()
            conn.close()


