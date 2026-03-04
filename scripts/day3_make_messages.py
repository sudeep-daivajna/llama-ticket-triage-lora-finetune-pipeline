import json
from pathlib import Path

SYSTEM_PROMPT_PATH = Path("configs/prompts/triage_system.txt")

IN_TRAIN = Path("data/train/train.jsonl")
IN_VAL = Path("data/train/val.jsonl")

OUT_DIR = Path("data/sft")
OUT_TRAIN = OUT_DIR / "train_messages.jsonl"
OUT_VAL = OUT_DIR / "val_messages.jsonl"

KEY_ORDER = ["type", "queue", "priority"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dumps_ordered_gold(gold: dict) -> str:
    # Force stable key order + compact JSON
    obj = {k: gold[k] for k in KEY_ORDER}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_user(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    return f"### Subject\n{subject}\n\n### Body\n{body}"


def convert_file(in_path: Path, out_path: Path, system_prompt: str):
    rows = []
    for ex in load_jsonl(in_path):
        ex_id = ex["id"]
        subject = ex["input"]["subject"]
        body = ex["input"]["body"]
        gold = ex["gold"]

        msg = {
            "id": ex_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_user(subject, body)},
                {"role": "assistant", "content": dumps_ordered_gold(gold)},
            ],
        }
        rows.append(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} | rows={len(rows):,}")


def main():
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing system prompt: {SYSTEM_PROMPT_PATH}")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    for p in [IN_TRAIN, IN_VAL]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    convert_file(IN_TRAIN, OUT_TRAIN, system_prompt)
    convert_file(IN_VAL, OUT_VAL, system_prompt)


if __name__ == "__main__":
    main()