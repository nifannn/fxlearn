import pymysql

try:
    import pandas as pd
    HAS_PANDAS = True
except:
    HAS_PANDAS = False

class MYSQLClient(object):
    """docstring for MYSQLConn"""
    def __init__(self, config):
        self._config = config
        self._conn = self._connect()

    def _connect(self):
        try:
            conn = pymysql.connect(**self._config)
            return conn
        except Exception as e:
            raise e

    def query(self, sql, args=None, return_df=False):
        with self._conn.cursor() as cur:
            if args is None:
                cur.execute(sql)
            else:
                cur.execute(sql, args)
            columns = [col[0] for col in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        if return_df:
            if HAS_PANDAS:
                return pd.DataFrame(results)
            print("Please install pandas ")
        return results

    def execute(self, sql, args=None):
        with self._conn.cursor() as cur:
            if args is None:
                cur.execute(sql)
            else:
                cur.execute(sql, args)
        self._conn.commit()

    def close(self):
        self._conn.close()