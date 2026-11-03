"""Rebuild monthly AIS SQLite databases as clustered WITHOUT ROWID copies.

The source databases are opened read-only.  Each target is first written as
``ais_YYYYMM.building.db`` and is renamed to ``ais_YYYYMM.db`` only after row
count, ship_info, sampled values, and SQLite quick-check validation succeeds.

The source table is scanned sequentially (``NOT INDEXED``), externally sorted
by ``MMSI, UTCtime``, and bulk inserted into a table whose primary key is the
same pair.  This avoids the highly fragmented table lookups observed in the
2025 databases while preserving every source row and value.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from typing import Iterable


AIS_COLUMNS = ("MMSI", "UTCtime", "Speed", "Longitude", "Latitude")


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _size_gb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**3
    except OSError:
        return 0.0


def _directory_size_gb(path: Path) -> float:
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total / 1024**3


class Reporter:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lock = threading.Lock()

    def write(self, message: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class Heartbeat:
    def __init__(self, reporter: Reporter, month: str, phase: str,
                 output_path: Path, temp_dir: Path, interval: float = 10.0):
        self.reporter = reporter
        self.month = month
        self.phase = phase
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.interval = interval
        self.started = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        spinner = "|/-\\"
        tick = 0
        while not self._stop.wait(self.interval):
            tick += 1
            free_gb = shutil.disk_usage(self.output_path.parent).free / 1024**3
            self.reporter.write(
                f"{spinner[tick % len(spinner)]} {self.month} [{self.phase}] "
                f"elapsed={_fmt_duration(time.time()-self.started)}  "
                f"output={_size_gb(self.output_path):.1f} GB  "
                f"temp={_directory_size_gb(self.temp_dir):.1f} GB  "
                f"output_free={free_gb:.1f} GB"
            )

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        status = "FAILED" if exc_type else "done"
        self.reporter.write(
            f"{self.month} [{self.phase}] {status}; "
            f"elapsed={_fmt_duration(time.time()-self.started)}  "
            f"output={_size_gb(self.output_path):.1f} GB"
        )


def _readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro&immutable=1"


def _source_row_count(con: sqlite3.Connection) -> int:
    row = con.execute(
        "SELECT stat FROM src.sqlite_stat1 "
        "WHERE tbl='ais_tracks' AND idx='idx_mmsi_time'"
    ).fetchone()
    if row and row[0]:
        return int(str(row[0]).split()[0])
    raise RuntimeError(
        "source sqlite_stat1/idx_mmsi_time is missing; run ANALYZE on source first"
    )


def _copy_ship_info(con: sqlite3.Connection) -> int:
    schema = con.execute(
        "SELECT sql FROM src.sqlite_master "
        "WHERE type='table' AND name='ship_info'"
    ).fetchone()
    if not schema or not schema[0]:
        raise RuntimeError("source ship_info schema is missing")
    con.execute(schema[0])
    before = con.total_changes
    con.execute("INSERT INTO ship_info SELECT * FROM src.ship_info")
    copied = con.total_changes - before
    con.commit()
    return int(copied)


def _sample_mmsi(con: sqlite3.Connection, limit: int = 3) -> list[int]:
    rows = con.execute(
        "SELECT MMSI FROM src.ais_tracks INDEXED BY idx_mmsi_time "
        "GROUP BY MMSI LIMIT ?", (limit,)
    ).fetchall()
    return [int(row[0]) for row in rows]


def _fingerprint(con: sqlite3.Connection, table: str, mmsi: int) -> tuple:
    # Scaled integer sums provide deterministic value checks without changing
    # the copied data.  Values remain well within SQLite signed 64-bit range.
    sql = f"""
        SELECT COUNT(*), MIN(UTCtime), MAX(UTCtime),
               COALESCE(SUM(CAST(ROUND(Speed * 1000.0) AS INTEGER)), 0),
               COALESCE(SUM(CAST(ROUND(Longitude * 1000000.0) AS INTEGER)), 0),
               COALESCE(SUM(CAST(ROUND(Latitude * 1000000.0) AS INTEGER)), 0)
        FROM {table} WHERE MMSI=?
    """
    return tuple(con.execute(sql, (mmsi,)).fetchone())


def _cleanup_building(path: Path) -> None:
    parent = path.parent.resolve()
    if path.resolve().parent != parent:
        raise RuntimeError(f"unsafe building path: {path}")
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _validate_ship_info(con: sqlite3.Connection) -> None:
    source_count = int(con.execute("SELECT COUNT(*) FROM src.ship_info").fetchone()[0])
    target_count = int(con.execute("SELECT COUNT(*) FROM ship_info").fetchone()[0])
    if source_count != target_count:
        raise RuntimeError(
            f"ship_info count mismatch: source={source_count}, target={target_count}"
        )
    left = int(con.execute(
        "SELECT COUNT(*) FROM (SELECT * FROM src.ship_info EXCEPT SELECT * FROM ship_info)"
    ).fetchone()[0])
    right = int(con.execute(
        "SELECT COUNT(*) FROM (SELECT * FROM ship_info EXCEPT SELECT * FROM src.ship_info)"
    ).fetchone()[0])
    if left or right:
        raise RuntimeError(f"ship_info content mismatch: source_only={left}, target_only={right}")


def rebuild_month(source: Path, final: Path, temp_dir: Path,
                  reporter: Reporter, *, force: bool, sample_count: int) -> None:
    month = source.stem.removeprefix("ais_")
    building = final.with_name(final.stem + ".building.db")
    verification = final.with_name(final.stem + ".verification.json")

    if final.exists() and not force:
        reporter.write(f"{month} already complete, skipping: {final}")
        return
    if final.exists() and force:
        raise RuntimeError(
            f"refusing to overwrite completed target in force mode; move it first: {final}"
        )
    _cleanup_building(building)

    source_size = _size_gb(source)
    reporter.write("=" * 78)
    reporter.write(f"START {month}: {source} ({source_size:.1f} GB)")
    reporter.write(f"TARGET: {final}")
    reporter.write(f"TEMP:   {temp_dir}")

    # TMP/TEMP are set before SQLite creates its external sorter files.
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)

    # ``uri=True`` must be enabled on the primary connection for SQLite to
    # interpret the read-only URI used by the attached source database on
    # Windows.
    build_uri = building.resolve().as_uri() + "?mode=rwc"
    con = sqlite3.connect(build_uri, uri=True, timeout=120)
    try:
        con.execute("PRAGMA page_size=8192")
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA locking_mode=EXCLUSIVE")
        con.execute("PRAGMA temp_store=FILE")
        con.execute("PRAGMA cache_size=-524288")
        con.execute("PRAGMA mmap_size=34359738368")
        con.execute("PRAGMA threads=4")
        escaped_temp = str(temp_dir).replace("'", "''").replace("\\", "/")
        try:
            con.execute(f"PRAGMA temp_store_directory='{escaped_temp}'")
        except sqlite3.DatabaseError:
            pass
        con.execute("ATTACH DATABASE ? AS src", (_readonly_uri(source),))
        try:
            con.execute("PRAGMA src.mmap_size=34359738368")
        except sqlite3.DatabaseError:
            pass

        source_rows = _source_row_count(con)
        ship_rows = _copy_ship_info(con)
        reporter.write(
            f"{month} source rows={source_rows:,}; ship_info rows={ship_rows:,}"
        )
        con.execute("""
            CREATE TABLE ais_tracks (
                MMSI INTEGER NOT NULL,
                UTCtime INTEGER NOT NULL,
                Speed REAL,
                Longitude REAL,
                Latitude REAL,
                PRIMARY KEY (MMSI, UTCtime)
            ) WITHOUT ROWID
        """)
        con.commit()

        plan = con.execute("""
            EXPLAIN QUERY PLAN
            SELECT MMSI, UTCtime, Speed, Longitude, Latitude
            FROM src.ais_tracks NOT INDEXED
            ORDER BY MMSI, UTCtime
        """).fetchall()
        reporter.write(f"{month} source plan: {plan}")
        reporter.write(
            f"{month} bulk rebuild begins: sequential source scan -> external sort "
            "-> clustered WITHOUT ROWID insert"
        )

        before = con.total_changes
        with Heartbeat(reporter, month, "SORT+BUILD", building, temp_dir):
            con.execute("BEGIN IMMEDIATE")
            con.execute("""
                INSERT INTO ais_tracks (MMSI, UTCtime, Speed, Longitude, Latitude)
                SELECT MMSI, UTCtime, Speed, Longitude, Latitude
                FROM src.ais_tracks NOT INDEXED
                ORDER BY MMSI, UTCtime
            """)
            con.execute("COMMIT")
        inserted = int(con.total_changes - before)
        reporter.write(f"{month} inserted rows={inserted:,}")
        if inserted != source_rows:
            raise RuntimeError(
                f"row-count mismatch: source={source_rows:,}, inserted={inserted:,}"
            )

        _validate_ship_info(con)
        samples = _sample_mmsi(con, max(1, sample_count))
        sample_results = []
        with Heartbeat(reporter, month, "SAMPLE-CHECK", building, temp_dir):
            for mmsi in samples:
                src_fp = _fingerprint(con, "src.ais_tracks", mmsi)
                dst_fp = _fingerprint(con, "ais_tracks", mmsi)
                if src_fp != dst_fp:
                    raise RuntimeError(
                        f"sample mismatch MMSI={mmsi}: source={src_fp}, target={dst_fp}"
                    )
                sample_results.append({"MMSI": mmsi, "fingerprint": list(src_fp)})

        # Qualify the pragma explicitly.  An unqualified quick_check also
        # inspects attached databases in this SQLite build; a damaged source
        # index must not make an already validated clustered target fail.
        with Heartbeat(reporter, month, "QUICK-CHECK", building, temp_dir):
            quick = str(con.execute("PRAGMA main.quick_check").fetchone()[0])
        if quick.lower() != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick}")

        target_plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM ais_tracks "
            "WHERE MMSI=? ORDER BY UTCtime", (samples[0],)
        ).fetchall()
        reporter.write(f"{month} target query plan: {target_plan}")
        con.execute("PRAGMA optimize")
        con.commit()
        con.execute("DETACH DATABASE src")
        con.close()
        con = None

        meta = {
            "month": month,
            "source": str(source),
            "target": str(final),
            "source_rows": source_rows,
            "inserted_rows": inserted,
            "ship_info_rows": ship_rows,
            "source_size_gb": source_size,
            "target_size_gb": _size_gb(building),
            "sqlite_quick_check": quick,
            "target_query_plan": target_plan,
            "samples": sample_results,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        verification.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(building, final)
        reporter.write(
            f"COMPLETE {month}: {final} ({_size_gb(final):.1f} GB); "
            f"verification={verification.name}"
        )
    except Exception:
        reporter.write(
            f"ERROR {month}: incomplete database retained for diagnosis: {building}"
        )
        raise
    finally:
        if con is not None:
            con.close()


def _parse_months(text: str) -> list[str]:
    months = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [m for m in months if len(m) != 6 or not m.isdigit()
               or not 1 <= int(m[-2:]) <= 12]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid YYYYMM values: {invalid}")
    return months


def _existing_sources(source_dir: Path, months: Iterable[str]) -> list[Path]:
    paths = [source_dir / f"ais_{month}.db" for month in months]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source databases: " + ", ".join(missing))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly rebuild AIS SQLite databases clustered by MMSI and UTCtime"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--months", type=str, required=True,
                        help="comma-separated YYYYMM values")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    temp_dir = args.temp_dir.resolve()
    months = _parse_months(args.months)
    sources = _existing_sources(source_dir, months)
    if source_dir == output_dir:
        raise SystemExit("source-dir and output-dir must differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cluster_rebuild.log"
    reporter = Reporter(log_path)

    source_gb = sum(_size_gb(path) for path in sources)
    output_free = shutil.disk_usage(output_dir).free / 1024**3
    temp_free = shutil.disk_usage(temp_dir).free / 1024**3
    max_source = max(_size_gb(path) for path in sources)
    reporter.write(
        f"PREFLIGHT months={months}; source={source_gb:.1f} GB; "
        f"output_free={output_free:.1f} GB; temp_free={temp_free:.1f} GB"
    )
    if output_free < source_gb * 1.05:
        raise SystemExit(
            f"insufficient output space: free={output_free:.1f} GB, "
            f"recommended>={source_gb*1.05:.1f} GB"
        )
    if temp_free < max_source * 1.20:
        raise SystemExit(
            f"insufficient temp space: free={temp_free:.1f} GB, "
            f"recommended>={max_source*1.20:.1f} GB"
        )
    if args.preflight_only:
        reporter.write("PREFLIGHT OK; no database was modified")
        return 0

    started = time.time()
    for source in sources:
        final = output_dir / source.name
        rebuild_month(
            source, final, temp_dir, reporter,
            force=bool(args.force), sample_count=max(1, int(args.sample_count)),
        )
    reporter.write(
        f"ALL REQUESTED MONTHS COMPLETE; elapsed={_fmt_duration(time.time()-started)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user; source databases were not modified.", flush=True)
        raise SystemExit(130)
