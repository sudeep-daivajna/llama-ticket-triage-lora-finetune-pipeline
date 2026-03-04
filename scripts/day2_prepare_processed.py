import re
import hashlib
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "tickets_en.parquet"

REQUIRED_COLS = ["subject", "body", "type", "queue", "priority", "language"]

ALLOWED_TYPE = {"Incident", "Request", "Problem", "Change"}
ALLOWED_PRIORITY = {"high", "medium", "low"}
ALLOWED_QUEUE = {
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Other",
}

RARE_QUEUES = {
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "General Inquiry",
    "Human Resources",
}


_whitespace_re = re.compile(r"\s+")


def normalize_text(x: object) -> str:
    """Lowercase + strip + collapse whitespace. No punctuation removal."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = s.strip().lower()
    s = _whitespace_re.sub(" ", s)
    return s


def make_id(subject: str, body: str) -> str:
    payload = f"{normalize_text(subject)}\n{normalize_text(body)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def validate_values(df: pd.DataFrame) -> None:
    bad_type = sorted(set(df["type"].dropna()) - ALLOWED_TYPE)
    bad_pri = sorted(set(df["priority"].dropna()) - ALLOWED_PRIORITY)
    bad_queue = sorted(set(df["queue"].dropna()) - ALLOWED_QUEUE)

    if bad_type or bad_pri or bad_queue:
        msg = []
        if bad_type:
            msg.append(f"Unexpected type values: {bad_type}")
        if bad_pri:
            msg.append(f"Unexpected priority values: {bad_pri}")
        if bad_queue:
            msg.append(f"Unexpected queue values: {bad_queue}")
        raise ValueError("Label validation failed.\n" + "\n".join(msg))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Find a CSV in data/raw (or change this to a fixed filename if you prefer)
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No .csv file found in data/raw/. Put the dataset there first.")
    if len(csv_files) > 1:
        print("Multiple CSVs found in data/raw/. Using the first one:")
        for p in csv_files:
            print(" -", p.name)
    raw_path = csv_files[0]

    df = pd.read_csv(raw_path)
    print(f"Loaded raw: {raw_path} | rows={len(df):,} cols={len(df.columns):,}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}\nFound columns: {list(df.columns)}")

    df = df[REQUIRED_COLS].copy()

    # English filter
    df = df[df["language"] == "en"].copy()
    print(f"After language=='en': rows={len(df):,}")

    # Fill missing text fields
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Basic cleanup for labels (strip only; don't change semantics)
    df["type"] = df["type"].astype(str).str.strip()
    df["queue"] = df["queue"].astype(str).str.strip()
    df.loc[df["queue"].isin(RARE_QUEUES), "queue"] = "Other"
    df["priority"] = df["priority"].astype(str).str.strip().str.lower()

    validate_values(df)

    # Stable id
    df["id"] = df.apply(lambda r: make_id(r["subject"], r["body"]), axis=1)

    # Detect conflicts: same content id but different labels
    label_cols = ["type", "queue", "priority"]
    conflicts = (
        df.groupby("id")[label_cols]
        .nunique(dropna=False)
        .reset_index()
    )
    conflict_ids = conflicts[
        (conflicts["type"] > 1) | (conflicts["queue"] > 1) | (conflicts["priority"] > 1)
    ]["id"].tolist()

    if conflict_ids:
        ex = df[df["id"].isin(conflict_ids)].head(10)[["id", "subject", "body"] + label_cols]
        raise ValueError(
            f"Found {len(conflict_ids)} conflicting duplicate IDs (same text, different labels).\n"
            f"Example rows:\n{ex.to_string(index=False)}"
        )

    # Deduplicate exact duplicates by id (safe because no conflicts)
    before = len(df)
    df = df.drop_duplicates(subset=["id"]).copy()
    after = len(df)
    print(f"After dedupe by id: rows={after:,} (dropped {before - after:,})")

    # Save processed (language column no longer needed)
    df_out = df[["id", "subject", "body", "type", "queue", "priority"]].copy()
    df_out.to_parquet(OUT_PATH, index=False)
    print(f"Saved processed parquet -> {OUT_PATH}")

    # Sanity-check distributions
    print("\nType distribution:")
    print(df_out["type"].value_counts(dropna=False))

    print("\nQueue distribution:")
    print(df_out["queue"].value_counts(dropna=False))

    print("\nPriority distribution:")
    print(df_out["priority"].value_counts(dropna=False))


if __name__ == "__main__":
    main()