from __future__ import annotations
import sqlite3

TW_COLUMNS = [
    "date", "close", "change_points", "turnover", "volume", "trades",
    "margin_balance_ntd", "margin_balance_change_ntd", "short_lots",
    "foreign_net_ntd", "trust_net_ntd", "dealer_net_ntd", "inst_total_net_ntd",
    "adv_count", "dec_count",
    "price_asof_taipei", "margin_asof_taipei", "inst_asof_taipei", "breadth_asof_taipei",
    "source_updated_at", "available_at_taipei", "quality_flags", "missing_fields",
]
US_COLUMNS = [
    "date", "spy_close", "spy_volume", "qqq_close", "qqq_volume",
    "iwm_close", "iwm_volume", "comp_close",
    "vix_close", "vix_open", "vix_high", "vix_low",
    "price_asof_taipei", "vix_asof_taipei",
    "source_updated_at", "available_at_taipei", "quality_flags", "missing_fields",
]
FINRA_COLUMNS = [
    "month", "debit_millions", "free_credit_cash_millions", "free_credit_margin_millions",
    "publish_date", "available_at_taipei", "source_updated_at", "quality_flags",
]

def _col_defs(cols: list[str], pk: str) -> str:
    parts = []
    for c in cols:
        if c == pk:
            parts.append(f"{c} TEXT PRIMARY KEY")
        elif c in {"date", "month"} or c.endswith("_taipei") or c in {
            "source_updated_at", "quality_flags", "missing_fields", "publish_date",
        }:
            parts.append(f"{c} TEXT")
        else:
            parts.append(f"{c} REAL")
    return ", ".join(parts)

class HistoryStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        c = self.conn
        c.execute(f"CREATE TABLE IF NOT EXISTS tw_daily ({_col_defs(TW_COLUMNS, 'date')})")
        c.execute(f"CREATE TABLE IF NOT EXISTS us_daily ({_col_defs(US_COLUMNS, 'date')})")
        c.execute(f"CREATE TABLE IF NOT EXISTS finra_monthly ({_col_defs(FINRA_COLUMNS, 'month')})")
        c.execute(
            "CREATE TABLE IF NOT EXISTS ingest_log "
            "(source TEXT, key TEXT, status TEXT, fetched_at TEXT, PRIMARY KEY(source, key))"
        )
        c.commit()

    def _upsert(self, table: str, cols: list[str], pk: str, row: dict) -> None:
        names = [c for c in cols if c in row]
        if pk not in names:
            raise ValueError(f"{table} upsert requires '{pk}'")
        placeholders = ", ".join("?" for _ in names)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c != pk)
        sql = (
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders}) "
            f"ON CONFLICT({pk}) DO UPDATE SET {updates}"
        ) if updates else (
            f"INSERT OR IGNORE INTO {table} ({', '.join(names)}) VALUES ({placeholders})"
        )
        self.conn.execute(sql, [row[c] for c in names])
        self.conn.commit()

    def upsert_tw_daily(self, row: dict) -> None:
        self._upsert("tw_daily", TW_COLUMNS, "date", row)

    def upsert_us_daily(self, row: dict) -> None:
        self._upsert("us_daily", US_COLUMNS, "date", row)

    def upsert_finra_monthly(self, row: dict) -> None:
        self._upsert("finra_monthly", FINRA_COLUMNS, "month", row)

    def _query(self, table: str, order_col: str, start, end) -> list[dict]:
        sql = f"SELECT * FROM {table}"
        clauses, params = [], []
        if start is not None:
            clauses.append(f"{order_col} >= ?"); params.append(start)
        if end is not None:
            clauses.append(f"{order_col} <= ?"); params.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_col} ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_tw_rows(self, start=None, end=None) -> list[dict]:
        return self._query("tw_daily", "date", start, end)

    def get_us_rows(self, start=None, end=None) -> list[dict]:
        return self._query("us_daily", "date", start, end)

    def get_finra_rows(self) -> list[dict]:
        return self._query("finra_monthly", "month", None, None)

    def mark_ingested(self, source: str, key: str, status: str = "ok") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ingest_log (source, key, status, fetched_at) "
            "VALUES (?, ?, ?, datetime('now'))", (source, key, status),
        )
        self.conn.commit()

    def is_ingested(self, source: str, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM ingest_log WHERE source=? AND key=? AND status='ok'",
            (source, key),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self.conn.close()
