import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonschema import Draft202012Validator

SYSTEM_PROMPT_PATH = Path("configs/prompts/triage_system.txt")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def load_schema(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# def eval_split(
#         name: str,
#         items
# )

def main():
    cfg_path = Path("configs/base.yaml")
    cfg = load_yaml(cfg_path)

    model_id = cfg["model"]["model_id"]
    max_seq_len = int(cfg["model"].get("max_seq_len", 2048))
    max_new_tokens = int(cfg["eval"].get("max_new_tokens", 64))

    eval_set_path = Path(cfg["eval"]["eval_set_path"])
    reg_set_path = Path(cfg["eval"]["regression_set_path"])
    schema_path = Path(cfg["eval"]["schema_path"])

    if not SYSTEM_PROMPT_PATH.exists():
         raise FileNotFoundError(f"Missing system prompt: {SYSTEM_PROMPT_PATH}")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    if not eval_set_path.exists():
        raise FileNotFoundError(f"Missing eval set: {eval_set_path}")
    if not reg_set_path.exists():
        raise FileNotFoundError(f"Missing regression set: {reg_set_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    
    validator = load_schema(schema_path)

    print(f"Config: {cfg_path}")
    print(f"Model:  {model_id}")
    print(f"SeqLen: {max_seq_len} | max_new_tokens={max_new_tokens}")
    print(f"Eval:   {eval_set_path}")
    print(f"Regr:   {reg_set_path}")
    print(f"Schema: {schema_path}")

    dtype = cfg["model"].get("dtype", "bf16")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    model.eval()

    eval_items = list(load_jsonl(eval_set_path))
    reg_items = list(load_jsonl(reg_set_path))

    out_dir = Path(cfg["run"].get("output_root", "runs"))/"baseline"/now_tag()
    eval_preds_path = out_dir / "preds_eval.jsonl"
    reg_preds_path = out_dir / "preds_regression.jsonl"

    print(f"\nSaving outputs to: {out_dir}")

    # Eval
    print("\n=== Running EVAL (has gold) ===")

    # m_eval = eval_split(
    #     "Eval",
    #     eval_items,
    #     model,
    #     tok,
    #     validator,
    #     system_prompt,
    #     max_seq_len,
    #     max_new_tokens,
    #     eval_preds_path,
    #     has_gold = True
    # )


if __name__ == "__main__":
    main()