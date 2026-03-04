from __future__ import annotations
import shutil, time
from pathlib import Path

def main(config_path: str):
    ts = time.strftime("%Y%m%d-%H%M")
    run_id = f"{ts}__BASE__schemajson__seed42"  # update later to read config
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "notes.md").write_text("# Notes\n\n- \n", encoding="utf-8")
    print("Created:", run_dir)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")