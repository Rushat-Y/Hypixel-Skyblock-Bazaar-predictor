
import argparse
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import sqlite3
import sys
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_session() -> requests.Session:
    """Build a requests Session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_session = make_session()

BAZAAR_URL  = "https://api.hypixel.net/v2/skyblock/bazaar"
ELECTION_URL = "https://api.hypixel.net/v2/resources/skyblock/election"
DEFAULT_DB   = "bazaar.db"
DEFAULT_INTERVAL = 60
ELECTION_TTL     = 300
PRINT_INTERVAL   = 300


def init_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             INTEGER NOT NULL,
            product_id     TEXT    NOT NULL,
            sellPrice      REAL,
            buyPrice       REAL,
            sellVolume     REAL,
            buyVolume      REAL,
            sellMovingWeek REAL,
            buyMovingWeek  REAL,
            spread         REAL,
            spreadPct      REAL,
            mayor          TEXT DEFAULT 'Unknown'
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_pid_ts
        ON snapshots (product_id, ts)
    """)
    con.commit()
    return con


def save_rows(con: sqlite3.Connection, rows: list):
    con.executemany("""
        INSERT INTO snapshots
          (ts, product_id, sellPrice, buyPrice, sellVolume, buyVolume,
           sellMovingWeek, buyMovingWeek, spread, spreadPct, mayor)
        VALUES
          (:ts, :product_id, :sellPrice, :buyPrice, :sellVolume, :buyVolume,
           :sellMovingWeek, :buyMovingWeek, :spread, :spreadPct, :mayor)
    """, rows)
    con.commit()


def get_stats(con: sqlite3.Connection) -> dict:
    total   = con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    items   = con.execute("SELECT COUNT(DISTINCT product_id) FROM snapshots").fetchone()[0]
    oldest  = con.execute("SELECT MIN(ts) FROM snapshots").fetchone()[0]
    return {"total": total, "items": items, "oldest_ms": oldest}


def fetch_mayor(current_mayor: str, last_fetch: float) -> tuple[str, float]:
    """Returns (mayor_string, last_fetch_time). Caches for ELECTION_TTL seconds."""
    if time.time() - last_fetch < ELECTION_TTL:
        return current_mayor, last_fetch
    try:
        r = _session.get(ELECTION_URL, timeout=8, verify=False)
        r.raise_for_status()
        em = r.json()
        if em.get("success") and "mayor" in em:
            name = em["mayor"]["name"]
            minister = em["mayor"].get("minister", {}).get("name", "")
            current_mayor = f"{name}+{minister}" if minister else name
    except Exception:
        pass
    return current_mayor, time.time()


def fetch_bazaar(mayor: str) -> list:
    """Fetches current bazaar snapshot. Returns list of row dicts."""
    r = _session.get(BAZAAR_URL, timeout=15, verify=False)
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError:
        snippet = r.text[:150].replace('\n', ' ')
        raise ValueError(f"Invalid JSON received from server. Payload: {snippet}...")

    if not isinstance(data, dict) or not data.get("success"):
        raise ValueError("Hypixel API returned success=false")

    ts   = int(data["lastUpdated"])
    rows = []
    for pid, product in data["products"].items():
        qs     = product.get("quick_status", {})
        sell_p = float(qs.get("sellPrice", 0))
        buy_p  = float(qs.get("buyPrice",  0))
        if sell_p <= 0 or buy_p <= 0:
            continue

        spread     = buy_p - sell_p
        spread_pct = spread / sell_p * 100

        rows.append({
            "ts":              ts,
            "product_id":      pid,
            "sellPrice":       sell_p,
            "buyPrice":        buy_p,
            "sellVolume":      float(qs.get("sellVolume",      0)),
            "buyVolume":       float(qs.get("buyVolume",       0)),
            "sellMovingWeek":  float(qs.get("sellMovingWeek",  0)),
            "buyMovingWeek":   float(qs.get("buyMovingWeek",   0)),
            "spread":          spread,
            "spreadPct":       spread_pct,
            "mayor":           mayor,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Bazaar Intel — Data Collector")
    parser.add_argument("--db",       default=DEFAULT_DB,       help="SQLite DB path")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, type=int, help="Poll interval (seconds)")
    args = parser.parse_args()

    print(f"[Collector] Starting — DB: {args.db}  Interval: {args.interval}s")
    print("[Collector] Press Ctrl+C to stop cleanly.\n")

    con  = init_db(args.db)
    mayor, last_election = "Unknown", 0.0
    polls = 0
    last_print = time.time()
    start_time = time.time()

    try:
        while True:
            poll_start = time.time()

            mayor, last_election = fetch_mayor(mayor, last_election)

            try:
                rows = fetch_bazaar(mayor)
                save_rows(con, rows)
                polls += 1
            except Exception as e:
                print(f"[Collector] Poll error: {e}")
                time.sleep(args.interval)
                continue

            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                stats    = get_stats(con)
                runtime  = str(datetime.timedelta(seconds=int(now - start_time)))
                oldest   = (
                    datetime.datetime.fromtimestamp(stats["oldest_ms"] / 1000).strftime("%Y-%m-%d %H:%M")
                    if stats["oldest_ms"] else "—"
                )
                print(
                    f"[{datetime.datetime.now():%H:%M:%S}] "
                    f"Polls: {polls} | "
                    f"Items: {stats['items']} | "
                    f"Snapshots: {stats['total']:,} | "
                    f"Oldest: {oldest} | "
                    f"Mayor: {mayor} | "
                    f"Runtime: {runtime}"
                )
                last_print = now

            elapsed = time.time() - poll_start
            sleep_for = max(0, args.interval - elapsed)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        stats = get_stats(con)
        print(f"\n[Collector] Stopped. {polls} polls collected. "
              f"{stats['total']:,} total snapshots across {stats['items']} items.")
        con.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
