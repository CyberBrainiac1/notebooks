#!/usr/bin/env python
"""Build an expanded dataset with targeted typo hard-cases and fact locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BOOSTER_GROUPS = [
    {
        "repeat": 3,
        "answer": "He said he is strong with SolidWorks and Onshape.",
        "questions": [
            "wat cad tools does he use",
            "wat cad tols does he use",
            "wat cad toools he use",
            "what cad sofware does he use",
            "what cad software does he use",
            "which cad platforms does he use",
            "which cad tools is he good with",
            "cad tools he uses?",
            "does he use solidworks and onshape",
            "is he good at solidworks and onshape",
        ],
    },
    {
        "repeat": 3,
        "answer": "He is based in the Evergreen area of San Jose, California.",
        "questions": [
            "where is he based",
            "wer iz he basd",
            "where he based",
            "where he basd",
            "whre is he based",
            "what city is he in",
            "which area in san jose is he from",
            "is he in evergreen san jose",
            "location?",
            "where does pranav live area wise",
        ],
    },
    {
        "repeat": 3,
        "answer": "I do not have that detail in Pranav's saved profile yet.",
        "questions": [
            "what is his birthday",
            "wats his bday",
            "what day is his birthday",
            "his birth date?",
            "what is his home address",
            "what is his phone number",
            "what is his social security number",
            "what is his bank account number",
        ],
    },
    {
        "repeat": 6,
        "answer": "His wheel setup uses a BTS7960 motor driver.",
        "questions": [
            "what motor driver does his wheel use",
            "wat motr drivr does his weel use",
            "which h bridge module does he use",
            "what exact driver name is used in the wheel setup",
            "is the wheel driver bts7960",
            "driver name for his sim wheel",
            "motor driver part number?",
            "what is the h-bridge model for his setup",
            "is it bts7960 or something else",
            "what driver board is on his wheel",
        ],
    },
    {
        "repeat": 4,
        "answer": "He uses 12V brushed DC planetary gearmotors around the 312 RPM output class.",
        "questions": [
            "what motor type does he use",
            "wat motors does he use with the wheel",
            "what are the wheel motors",
            "is he using brushed planetary motors",
            "what rpm class did he mention for motors",
            "which motors are paired with bts7960",
            "what motor class details did he share",
            "what motors for summed torque setup",
        ],
    },
    {
        "repeat": 6,
        "answer": "His exact rule is CPR = PPR x 4.",
        "questions": [
            "how does he define cpr from ppr",
            "how he defne cpr frm ppr",
            "what is his exact cpr formula",
            "is cpr ppr times 4",
            "does he use cpr equals ppr x4",
            "what quadrature rule did he lock in",
            "cpr formula?",
            "cpr from ppr rule",
            "is it x4 or divided",
            "what does he use for quadrature decoding",
        ],
    },
    {
        "repeat": 6,
        "answer": "No. He uses CPR = PPR x 4, not CPR = PPR x 4/3.",
        "questions": [
            "is his cpr rule divided by 3",
            "does he use ppr x 4/3",
            "is the formula cpr = ppr x 4 over 3",
            "does he divide by 3 in cpr",
            "cpr equals ppr x 4/3?",
            "is it x4/3 or x4",
            "is cpr ppr x four thirds",
            "do he use 4/3 multiplier",
        ],
    },
]


INSTRUCTION = (
    "Handle misspellings and short text-style prompts while using only saved profile facts."
)


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["instruction"].strip().lower(),
            row["input"].strip().lower(),
            row["output"].strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build expanded dataset with typo hard-cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pranav_profile_qa_v2.jsonl"),
        help="Base dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pranav_profile_qa_v4.jsonl"),
        help="Expanded dataset output path.",
    )
    parser.add_argument(
        "--repeat-scale",
        type=int,
        default=1,
        help="Global multiplier for booster group repeat counts.",
    )
    args = parser.parse_args()

    rows = dedupe_rows(load_rows(args.input))
    base_count = len(rows)
    booster_rows: list[dict[str, str]] = []
    for group in BOOSTER_GROUPS:
        answer = str(group["answer"])
        questions = group["questions"]
        repeat = int(group["repeat"]) * max(1, args.repeat_scale)
        for _ in range(repeat):
            for question in questions:
                booster_rows.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": str(question),
                        "output": answer,
                    }
                )

    rows.extend(booster_rows)

    write_rows(args.output, rows)
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"(base={base_count}, booster={len(booster_rows)})"
    )


if __name__ == "__main__":
    main()
