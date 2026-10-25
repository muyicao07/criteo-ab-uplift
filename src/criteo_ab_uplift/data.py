"""Verified public-data download, full SQL audit and disk-backed arrays.

Every analysis row is retained. CSV parsing uses bounded chunks, while the
float64 cache preserves precision for the full-data regression calculation.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import uuid

import duckdb
import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "visit", "conversion", "exposure"]
REQUIRED_COLUMNS = FEATURES + BINARY_COLUMNS
EXPECTED_ROWS = 13_979_592
CACHE_VERSION = 1
DATA_REVISION = "82811785048bb633de2d55c02bab4e57066e6423"
DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/"
    f"{DATA_REVISION}/criteo-research-uplift-v2.1.csv.gz"
)
DATA_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Kept public and as an alias for callers recording provenance.
_sha256 = sha256_file


def _verify_download(path: Path) -> Path:
    actual = sha256_file(path)
    if actual != DATA_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {DATA_SHA256}, got {actual}. "
            "The existing file was preserved; select a different destination."
        )
    return path


def download_data(destination: Path) -> Path:
    """Download the pinned official release, with progress and hash validation."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        print("Verifying existing Criteo download...", flush=True)
        return _verify_download(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "criteo-ab-uplift/1.0"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            received = 0
            last_progress = time.monotonic()
            print("Downloading the official Criteo v2.1 dataset...", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                total = response.headers.get("Content-Length")
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if time.monotonic() - last_progress >= 10:
                        total_text = f" / {int(total) / 2**30:.2f} GiB" if total else " GiB"
                        print(f"  downloaded {received / 2**30:.2f}{total_text}", flush=True)
                        last_progress = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded file failed its pinned SHA-256 check.")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return _verify_download(destination)
        print(f"Download complete; SHA-256 verified ({received / 2**30:.2f} GiB).", flush=True)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_header(path: Path) -> None:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    duplicates = sorted({name for name in header if header.count(name) > 1})
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing or duplicates:
        raise ValueError(f"Invalid CSV header: missing={missing}; duplicate={duplicates}.")


def _sql(filename: str) -> str:
    path = SQL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"SQL source missing: {path}. Install from this repository with pip install -e .")
    return path.read_text(encoding="utf-8")


def _audit_csv(path: Path, expected_rows: int, threads: int) -> dict:
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit = '2GB'")
        con.execute("SET threads = ?", [threads])
        literal = "'" + str(path).replace("'", "''") + "'"
        con.execute(
            "CREATE VIEW raw AS SELECT * FROM read_csv_auto("
            f"{literal}, header=true, all_varchar=true, strict_mode=true, "
            "ignore_errors=false, null_padding=false)"
        )
        casts = ", ".join(f'TRY_CAST("{column}" AS DOUBLE) AS "{column}"' for column in REQUIRED_COLUMNS)
        con.execute(f"CREATE VIEW data AS SELECT {casts} FROM raw")
        print("Auditing every row with SQL (row count and value checks)...", flush=True)
        validation = con.execute(_sql("01_validate.sql")).fetchone()
        n_total = int(validation[0])
        if n_total != expected_rows:
            raise ValueError(
                f"Dataset row count is {n_total:,}; exactly {expected_rows:,} rows are required. "
                "Stopped before analysis or model fitting."
            )
        invalid = {column: int(value) for column, value in zip(REQUIRED_COLUMNS, validation[1:]) if value}
        if invalid:
            raise ValueError(f"Invalid values: {invalid}; features must be finite, indicators exactly 0/1.")
        print("Computing full-data group statistics and covariate balance...", flush=True)
        rows = con.execute(_sql("02_group_statistics.sql")).fetchall()
        if len(rows) != 2 or [r[0] for r in rows] != [0, 1]:
            raise ValueError("The dataset must contain both treatment groups.")
        keys = ["treatment", "n", "visit_sum", "conversion_sum", "exposure_sum",
                "visit_exposure_sum", "conversion_exposure_sum"]
        groups = [dict(zip(keys, [int(value) for value in row[:7]])) for row in rows]
        if min(group["n"] for group in groups) < 2:
            raise ValueError("Both groups require at least two observations.")
        n0, n1 = (group["n"] for group in groups)
        balance, means, stds = [], [], []
        for index, feature in enumerate(FEATURES):
            mean0, std0 = map(float, rows[0][7 + 2 * index:9 + 2 * index])
            mean1, std1 = map(float, rows[1][7 + 2 * index:9 + 2 * index])
            pooled = math.sqrt((std0**2 + std1**2) / 2)
            smd = (mean1 - mean0) / pooled if pooled else (0.0 if mean1 == mean0 else None)
            balance.append({"feature": feature, "mean_0": mean0, "mean_1": mean1,
                            "std_0": std0, "std_1": std1, "smd": smd})
            mean = (n0 * mean0 + n1 * mean1) / n_total
            variance = ((n0 - 1) * std0**2 + (n1 - 1) * std1**2
                        + n0 * (mean0 - mean)**2 + n1 * (mean1 - mean)**2) / (n_total - 1)
            means.append(mean)
            stds.append(math.sqrt(max(0.0, variance)))
        observed_joint = {tuple(map(int, row[:3])): int(row[3])
                          for row in con.execute(_sql("03_joint_counts.sql")).fetchall()}
        joint = [{"treatment": treatment, "visit": visit, "conversion": conversion,
                  "n": observed_joint.get((treatment, visit, conversion), 0)}
                 for treatment in (0, 1) for visit in (0, 1) for conversion in (0, 1)]
        max_smd = max(abs(item["smd"]) for item in balance if item["smd"] is not None)
        return {
            "n_total": n_total, "expected_rows": expected_rows,
            "treatment_fraction": n1 / n_total,
            "control_exposure_count": groups[0]["exposure_sum"],
            "control_exposure_rate": groups[0]["exposure_sum"] / n0,
            "groups": groups, "joint_counts": joint, "balance": balance,
            "feature_means": means, "feature_stds": stds,
            "max_absolute_smd": max_smd,
            "balance_threshold": 0.1,
            "balance_holds": all(item["smd"] is not None and abs(item["smd"]) < 0.1 for item in balance),
            "validation": {"invalid_values": 0, "row_count_matches": True},
        }
    except duckdb.Error as exc:
        raise ValueError(f"Full-file SQL audit failed for {path}: {exc}") from exc
    finally:
        con.close()


def _cached_audit(cache: Path, digest: str, expected_rows: int) -> dict | None:
    try:
        audit = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        if (not audit.get("complete") or audit.get("cache_version") != CACHE_VERSION
                or audit.get("data_sha256") != digest or audit.get("n_total") != expected_rows):
            return None
        expected = {"X.npy": ((expected_rows, 12), np.dtype("float32")),
                    "X64.npy": ((expected_rows, 12), np.dtype("float64")),
                    "labels.npy": ((expected_rows, 4), np.dtype("uint8"))}
        for name, (shape, dtype) in expected.items():
            array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                return None
            del array
        audit["cache_dir"] = str(cache)
        return audit
    except (OSError, ValueError, KeyError, TypeError):
        return None


def prepare_data(csv_path: Path, cache_dir: Path, seed: int = 2026, threads: int = 4,
                 expected_rows: int = EXPECTED_ROWS) -> dict:
    """Audit the complete dataset and atomically prepare all-row memory maps.

    ``expected_rows`` exists for developer verification on tiny fixtures. The
    production entry point always supplies EXPECTED_ROWS and verifies SHA-256.
    ``seed`` is recorded for provenance; no rows are sampled by this function.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows < 4:
        raise ValueError("expected_rows must be an integer of at least four.")
    _check_header(csv_path)
    print("Fingerprinting the source data...", flush=True)
    digest = sha256_file(csv_path)
    existing = _cached_audit(cache_dir, digest, expected_rows)
    if existing is not None:
        print(f"Reusing verified all-row cache ({expected_rows:,} rows).", flush=True)
        existing["data_path"] = str(csv_path)
        return existing
    if cache_dir.exists() and any(cache_dir.iterdir()) and not (cache_dir / "metadata.json").is_file():
        raise ValueError(f"Refusing to replace a nonempty, unrecognized cache directory: {cache_dir}")
    audit = _audit_csv(csv_path, expected_rows, threads)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".building-", dir=cache_dir.parent))
    try:
        n = audit["n_total"]
        x = np.lib.format.open_memmap(temporary / "X.npy", mode="w+", dtype="float32", shape=(n, 12))
        x64 = np.lib.format.open_memmap(temporary / "X64.npy", mode="w+", dtype="float64", shape=(n, 12))
        labels = np.lib.format.open_memmap(temporary / "labels.npy", mode="w+", dtype="uint8", shape=(n, 4))
        start = 0
        print("Writing all rows to disk-backed feature and label arrays...", flush=True)
        for chunk in pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS,
                                 dtype={column: "float64" for column in REQUIRED_COLUMNS},
                                 chunksize=250_000, float_precision="round_trip"):
            end = start + len(chunk)
            if end > n:
                raise ValueError("The source row count changed while preparing data.")
            features = chunk[FEATURES].to_numpy(dtype="float64", copy=False)
            indicators = chunk[BINARY_COLUMNS].to_numpy(dtype="float64", copy=False)
            if not np.isfinite(features).all() or not np.isin(indicators, (0.0, 1.0)).all():
                raise ValueError("The source values changed after the SQL audit.")
            x64[start:end] = features
            x[start:end] = features
            labels[start:end] = indicators
            start = end
            if start % 2_000_000 == 0 or start == n:
                print(f"  retained {start:,} / {n:,} rows", flush=True)
        if start != n:
            raise ValueError(f"Expected {n:,} rows but materialized {start:,} rows.")
        x.flush()
        x64.flush()
        labels.flush()
        del x, x64, labels
        # Catch source changes occurring during the independent SQL/CSV passes.
        if sha256_file(csv_path) != digest:
            raise ValueError("The source file changed during preparation. Retry with an immutable download.")
        audit.update({"data_path": str(csv_path), "data_sha256": digest,
                      "data_url": DATA_URL, "data_revision": DATA_REVISION,
                      "cache_dir": str(cache_dir), "cache_version": CACHE_VERSION,
                      "seed": int(seed), "complete": True,
                      "feature_columns": FEATURES, "label_columns": BINARY_COLUMNS,
                      "materialization": "all rows, original file order; no subsampling"})
        (temporary / "metadata.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        backup = None
        if cache_dir.exists():
            backup = cache_dir.with_name(cache_dir.name + ".old-" + uuid.uuid4().hex)
            os.replace(cache_dir, backup)
        try:
            os.replace(temporary, cache_dir)
        except BaseException:
            if backup is not None:
                os.replace(backup, cache_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return audit
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
