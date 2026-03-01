import os
import getpass
import ibis

def wrds_connect(wrds_id=None, use_local=False):
    if use_local:
        user = getpass.getuser()
        return ibis.postgres.connect(
            host="localhost",
            port=5432,
            user=user,
            database=user,
        )
    else:
        if not wrds_id:
            wrds_id = os.getenv("WRDS_ID")
        return ibis.postgres.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            user=wrds_id,
            database="wrds",
        )

def wrds_table(db, table, schema):
    return db.table(table, database=schema)
