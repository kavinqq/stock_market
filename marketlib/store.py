from __future__ import annotations
import sqlite3
from typing import Optional

TW_COLUMNS = [
    "date", "close", "change_points", "turnover", "volume", "trades",
    "margin_balance_ntd", "margin_balance_change_ntd", "short_lots",
    "foreign_net_ntd", "trust_net_ntd", "dealer_net_ntd", "inst_total_net_ntd",
    "adv_count", "dec_count",
    "tx_foreign_oi_net_lots", "tx_inst_oi_net_lots",
    "price_asof_taipei", "margin_asof_taipei", "inst_asof_taipei", "breadth_asof_taipei",
    "taifex_asof_taipei",
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

STOCK_DAILY_COLUMNS = [
    "stock_no", "date", "open", "high", "low", "close", "volume",
    "foreign_net_lots", "trust_net_lots", "dealer_net_lots", "inst_total_net_lots",
    "price_asof_taipei", "chip_asof_taipei", "available_at_taipei",
    "quality_flags", "missing_fields",
]

STOCK_VALUATION_COLUMNS = [
    "stock_no", "date", "pe", "pb", "dividend_yield",
    "asof_taipei", "quality_flags",
]

STOCK_MONTHLY_REVENUE_COLUMNS = [
    "stock_no", "month", "revenue_thousand", "yoy_pct", "mom_pct", "cum_yoy_pct",
    "asof_taipei", "quality_flags",
]

_TEXT_COLS = {"stock_no", "date", "month", "source_updated_at", "quality_flags",
              "missing_fields", "publish_date"}

def _col_type_simple(c: str) -> str:
    if c in {"date", "month"} or c.endswith("_taipei") or c in {
        "source_updated_at", "quality_flags", "missing_fields", "publish_date",
    }:
        return "TEXT"
    return "REAL"

def _col_defs(cols: list[str], pk: str) -> str:
    parts = []
    for c in cols:
        if c == pk:
            parts.append(f"{c} TEXT PRIMARY KEY")
        else:
            parts.append(f"{c} {_col_type_simple(c)}")
    return ", ".join(parts)

def _col_defs_composite(cols: list[str], pk_cols: list[str]) -> str:
    """Build column definitions for a table with a composite primary key."""
    parts = []
    for c in cols:
        if c in _TEXT_COLS or c.endswith("_taipei") or c.endswith("_asof"):
            parts.append(f"{c} TEXT")
        else:
            parts.append(f"{c} REAL")
    parts.append(f"PRIMARY KEY({', '.join(pk_cols)})")
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
        c.execute(
            f"CREATE TABLE IF NOT EXISTS stock_daily "
            f"({_col_defs_composite(STOCK_DAILY_COLUMNS, ['stock_no', 'date'])})"
        )
        c.execute(
            f"CREATE TABLE IF NOT EXISTS stock_valuation "
            f"({_col_defs_composite(STOCK_VALUATION_COLUMNS, ['stock_no', 'date'])})"
        )
        c.execute(
            f"CREATE TABLE IF NOT EXISTS stock_monthly_revenue "
            f"({_col_defs_composite(STOCK_MONTHLY_REVENUE_COLUMNS, ['stock_no', 'month'])})"
        )
        # 對既有 DB 補上新增欄位(CREATE TABLE IF NOT EXISTS 不會動已存在的表)。
        self._add_missing_columns("tw_daily", TW_COLUMNS)
        self._add_missing_columns("us_daily", US_COLUMNS)
        c.commit()

    def _add_missing_columns(self, table: str, cols: list[str]) -> None:
        existing = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in cols:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {_col_type_simple(col)}")

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

    def _upsert_pk(self, table: str, cols: list[str], pk_cols: list[str], row: dict) -> None:
        """Composite-PK upsert; partial-update preserves columns not present in row."""
        names = [c for c in cols if c in row]
        missing_pks = [pk for pk in pk_cols if pk not in names]
        if missing_pks:
            raise ValueError(f"{table} upsert requires PK columns: {missing_pks}")
        placeholders = ", ".join("?" for _ in names)
        pk_set = set(pk_cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c not in pk_set)
        pk_conflict = ", ".join(pk_cols)
        sql = (
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders}) "
            f"ON CONFLICT({pk_conflict}) DO UPDATE SET {updates}"
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

    def upsert_stock_daily(self, row: dict) -> None:
        self._upsert_pk("stock_daily", STOCK_DAILY_COLUMNS, ["stock_no", "date"], row)

    def upsert_stock_valuation(self, row: dict) -> None:
        self._upsert_pk("stock_valuation", STOCK_VALUATION_COLUMNS, ["stock_no", "date"], row)

    def upsert_stock_monthly_revenue(self, row: dict) -> None:
        self._upsert_pk("stock_monthly_revenue", STOCK_MONTHLY_REVENUE_COLUMNS, ["stock_no", "month"], row)

    def _query_stock(self, table: str, stock_no: str, order_col: str,
                     start: Optional[str] = None, end: Optional[str] = None) -> list[dict]:
        sql = f"SELECT * FROM {table} WHERE stock_no = ?"
        params: list = [stock_no]
        if start is not None:
            sql += f" AND {order_col} >= ?"; params.append(start)
        if end is not None:
            sql += f" AND {order_col} <= ?"; params.append(end)
        sql += f" ORDER BY {order_col} ASC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_stock_daily(self, stock_no: str, start: Optional[str] = None,
                        end: Optional[str] = None) -> list[dict]:
        return self._query_stock("stock_daily", stock_no, "date", start, end)

    def get_stock_valuation(self, stock_no: str, start: Optional[str] = None,
                            end: Optional[str] = None) -> list[dict]:
        return self._query_stock("stock_valuation", stock_no, "date", start, end)

    def get_stock_monthly_revenue(self, stock_no: str) -> list[dict]:
        return self._query_stock("stock_monthly_revenue", stock_no, "month")

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
