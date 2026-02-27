#!/usr/bin/env python
"""Evaluate a local Pranav LoRA adapter on clean + misspelled profile prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are PranavProfileGPT, a profile-grounded assistant.\n"
    "Rules:\n"
    "- Use only known profile facts.\n"
    "- If unknown, say you do not have that detail yet.\n"
    "- Do not invent personal details."
)


PROMPT_TEMPLATE = """{system_prompt}

### Retrieved Memory:
- No relevant prior memory found.

### Instruction:
Answer the user question about Pranav's profile using facts only.

### Input:
{user_input}

### Response:
"""


SMS_MAP = {
    "what": "wat",
    "where": "wer",
    "which": "wich",
    "does": "doez",
    "motor": "motr",
    "driver": "drivr",
    "wheel": "weel",
    "define": "defne",
    "from": "frm",
    "birthday": "bday",
    "software": "softwere",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9']+")


def typo_token(token: str, rng: random.Random) -> str:
    if len(token) < 4:
        return token
    if rng.random() < 0.4:
        return token
    idx = rng.randint(1, len(token) - 2)
    if rng.random() < 0.5:
        return token[:idx] + token[idx + 1 :]
    return token[:idx] + token[idx] + token[idx:]


def noisy_variant(text: str, rng: random.Random) -> str:
    out: List[str] = []
    for tok in TOKEN_RE.findall(text):
        if re.fullmatch(r"[A-Za-z0-9']+", tok):
            low = tok.lower()
            low = SMS_MAP.get(low, low)
            low = typo_token(low, rng)
            out.append(low)
        else:
            out.append(tok)
    return "".join(out).strip().rstrip("?")


def has_terms(text: str, terms: Sequence[str]) -> bool:
    low = text.lower()
    return all(term in low for term in terms)


def check_motor_driver(text: str) -> bool:
    low = text.lower()
    if "bts7960" not in low:
        return False
    return not any(b in low for b in ["bts73", "bts78", "bts 73", "bts 78"])


def check_cpr_formula(text: str) -> bool:
    low = text.lower()
    if "cpr" not in low or "ppr" not in low:
        return False
    has_x4 = any(token in low for token in ["x 4", "x4", "*4", "* 4"])
    if not has_x4:
        return False
    return not any(b in low for b in ["4/3", "x 4/3", "x4/3", "*4/3"])


@dataclass
class Case:
    name: str
    question: str
    validator: Callable[[str], bool]


CASES: List[Case] = [
    Case("preferred_name", "What is his preferred name?", lambda t: has_terms(t, ["superman"])),
    Case("full_name", "What is his full name?", lambda t: has_terms(t, ["pranav emmadi"])),
    Case("location", "Where is he based?", lambda t: has_terms(t, ["evergreen", "san jose"])),
    Case("ftc_team", "What FTC team is he on?", lambda t: has_terms(t, ["evergreen dragons"])),
    Case("frc_team", "What FRC team did he join?", lambda t: has_terms(t, ["2854", "prototypes"])),
    Case("cad", "What CAD software does he use?", lambda t: has_terms(t, ["solidworks", "onshape"])),
    Case("motor_driver", "What motor driver does his wheel use?", check_motor_driver),
    Case(
        "motor_type",
        "What motor type does he use with that driver?",
        lambda t: has_terms(t, ["12v", "brushed", "planetary"]),
    ),
    Case("cpr_formula", "How does he define CPR from PPR?", check_cpr_formula),
    Case(
        "unknown_birthday",
        "What is his birthday?",
        lambda t: has_terms(t, ["do not have that detail"]),
    ),
    Case(
        "unknown_phone",
        "What is his phone number?",
        lambda t: has_terms(t, ["do not have that detail"]),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Pranav LoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/pranav_colab_3b/pranav_lora"),
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument(
        "--min-overall-pass",
        type=float,
        default=0.90,
        help="Minimum overall pass ratio required to return success.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_dir}")

    rng = random.Random(args.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_dir),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clean_total = 0
    clean_pass = 0
    noisy_total = 0
    noisy_pass = 0
    failures: List[Dict[str, str]] = []
    rows: List[Dict[str, object]] = []

    for case in CASES:
        variants = [("clean", case.question), ("noisy", noisy_variant(case.question, rng))]
        for variant_type, question in variants:
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                user_input=question,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
            )
            text = tokenizer.batch_decode(output)[0]
            answer = text.split("### Response:")[-1].strip()
            if "<|eot_id|>" in answer:
                answer = answer.split("<|eot_id|>")[0].strip()

            ok = case.validator(answer)
            rows.append(
                {
                    "case": case.name,
                    "variant": variant_type,
                    "question": question,
                    "answer": answer,
                    "pass": ok,
                }
            )
            if variant_type == "clean":
                clean_total += 1
                clean_pass += int(ok)
            else:
                noisy_total += 1
                noisy_pass += int(ok)
            if not ok:
                failures.append(
                    {
                        "case": case.name,
                        "variant": variant_type,
                        "question": question,
                        "answer": answer,
                    }
                )

    overall_pass = clean_pass + noisy_pass
    overall_total = clean_total + noisy_total
    overall_ratio = overall_pass / max(1, overall_total)
    print(f"Adapter: {args.adapter_dir}")
    print(f"Clean: {clean_pass}/{clean_total}")
    print(f"Noisy: {noisy_pass}/{noisy_total}")
    print(f"Overall: {overall_pass}/{overall_total} ({overall_ratio:.3f})")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"- [{f['variant']}] {f['case']}: {f['question']}")
            print(f"  -> {f['answer']}")
    else:
        print("\nAll checks passed.")

    if args.report:
        payload = {
            "adapter": str(args.adapter_dir),
            "clean": {"pass": clean_pass, "total": clean_total},
            "noisy": {"pass": noisy_pass, "total": noisy_total},
            "overall": {
                "pass": overall_pass,
                "total": overall_total,
                "ratio": overall_ratio,
            },
            "failures": failures,
            "rows": rows,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report: {args.report}")

    if overall_ratio < args.min_overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
