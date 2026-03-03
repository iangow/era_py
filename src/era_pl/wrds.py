def wrds_connect(wrds_id=None, use_local=False):
    raise NotImplementedError(
        "era_pl does not provide PostgreSQL/WRDS connections. "
        "Use local parquet workflows with events.load_parquet()."
    )


def wrds_table(db, table, schema):
    raise NotImplementedError(
        "era_pl does not provide PostgreSQL/WRDS table access."
    )
