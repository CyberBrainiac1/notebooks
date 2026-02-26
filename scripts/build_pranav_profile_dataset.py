#!/usr/bin/env python
"""Build an instruction-tuning dataset from Pranav's saved profile facts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


INSTRUCTION_POOL = [
    "Answer using only Pranav's saved profile facts. If unknown, say you do not have that detail.",
    "You are a profile assistant for Pranav Emmadi. Be concise, factual, and avoid making up details.",
    "Respond with profile-grounded information only. Do not invent personal data.",
]


UNKNOWN_OUTPUT = (
    "I do not have that detail in Pranav's saved profile yet."
)


QA_BANK: List[Dict[str, object]] = [
    {
        "questions": [
            "What is his preferred name?",
            "What should we call him?",
            "What nickname does he prefer?",
        ],
        "answer": "His preferred name is superman.",
    },
    {
        "questions": [
            "What is his full name?",
            "What name should be used in formal emails?",
        ],
        "answer": "His full name is Pranav Emmadi.",
    },
    {
        "questions": [
            "Where is he based?",
            "What city area does he mention?",
        ],
        "answer": "He is based in the Evergreen area of San Jose, California.",
    },
    {
        "questions": [
            "What is his student level?",
            "What grade level did he mention?",
        ],
        "answer": "He described himself as an advanced 9th grader and student.",
    },
    {
        "questions": [
            "What is his overall personality toward projects?",
            "How does he approach learning?",
        ],
        "answer": (
            "He is hands-on and maker-oriented. He likes to build fast, test, and iterate."
        ),
    },
    {
        "questions": [
            "What FTC team is he on?",
            "What is his FTC team name?",
        ],
        "answer": "He is on FTC team Evergreen Dragons.",
    },
    {
        "questions": [
            "What FRC team did he join?",
            "What is his FRC team and role direction?",
        ],
        "answer": "He joined FRC Team 2854 Prototypes and described himself as mechanical-focused.",
    },
    {
        "questions": [
            "What are his leadership goals in robotics?",
            "What are his timeline goals for team leadership?",
        ],
        "answer": (
            "His goals are mechanical lead by 10th grade and team captain by 11th grade."
        ),
    },
    {
        "questions": [
            "What engineering direction does he want long term?",
            "What career mix does he want?",
        ],
        "answer": (
            "He wants to become a mechanical engineer while also getting strong in computer science to bridge software and hardware."
        ),
    },
    {
        "questions": [
            "What major paper topic is he working on?",
            "What was his robotic hand research focus?",
        ],
        "answer": (
            "He worked on a DIY robotic hand research paper focused on replicating human hand movement, including design, mechanics, and control."
        ),
    },
    {
        "questions": [
            "What is the saved title of his robotic hand paper?",
            "What was the specific paper title?",
        ],
        "answer": (
            "The saved title is: Engineering a DIY Robotic Hand: Design, Mechanics, and Control."
        ),
    },
    {
        "questions": [
            "What other robotics projects has he mentioned?",
            "Give examples of his recurring projects.",
        ],
        "answer": (
            "He has mentioned a robotic arm, an FTC shooter project, and a DIY sim racing steering wheel build."
        ),
    },
    {
        "questions": [
            "What games got him into sim racing?",
            "Which sim racing games did he mention?",
        ],
        "answer": (
            "He got into sim racing through BeamNG and also started playing Assetto Corsa."
        ),
    },
    {
        "questions": [
            "What microcontroller does he use for the wheel?",
            "What is the main wheel controller board?",
        ],
        "answer": "His main microcontroller is an Arduino Leonardo.",
    },
    {
        "questions": [
            "What motor driver does his wheel setup use?",
            "Which H-bridge module did he mention?",
        ],
        "answer": "His wheel setup uses a BTS7960 motor driver.",
    },
    {
        "questions": [
            "What encoder type is in his wheel stack?",
            "What type of encoder does he use?",
        ],
        "answer": "He uses a quadrature incremental encoder with A/B channels.",
    },
    {
        "questions": [
            "What is his locked-in encoder rule?",
            "How does he define CPR from PPR?",
        ],
        "answer": "He explicitly uses CPR = PPR x 4 for quadrature decoding.",
    },
    {
        "questions": [
            "What wheel firmware/control routes did he explore?",
            "What software options has he asked about?",
        ],
        "answer": (
            "He explored EMC Lite wiring/config, asked about EMC Pro availability, compared FFBeast ideas, and asked for an EMC Lite-like desktop tuning app."
        ),
    },
    {
        "questions": [
            "Describe his current dual-motor plan.",
            "What is his in-progress two-motor architecture?",
        ],
        "answer": (
            "He is testing two 12V brushed planetary gearmotors geared into one shared shaft so torque adds together, with both motors commanded at the same time."
        ),
    },
    {
        "questions": [
            "What motor class details did he share?",
            "What are the wheel motor characteristics?",
        ],
        "answer": (
            "He described 12V brushed DC planetary gearmotors around the 312 RPM output class."
        ),
    },
    {
        "questions": [
            "What encoder count values did he reference?",
            "What PPR and CPR numbers did he mention?",
        ],
        "answer": (
            "He referenced 537.7 PPR at the output shaft, which is about 2150.8 CPR after x4 quadrature decoding."
        ),
    },
    {
        "questions": [
            "What encoder angle bugs has he seen?",
            "What wrong angle readings did he report?",
        ],
        "answer": (
            "He reported cases where expected 180 degrees showed about 379 degrees, and later another case where 180 degrees showed about 21 degrees."
        ),
    },
    {
        "questions": [
            "What pedal mapping problems did he hit?",
            "What control-axis issues were reported?",
        ],
        "answer": (
            "He reported reversed controls at times, throttle Z-axis issues, brake Y-axis issues, and Assetto Corsa not accepting a brake button bind in one case."
        ),
    },
    {
        "questions": [
            "How is his brake button wired?",
            "What brake input logic level did he mention?",
        ],
        "answer": (
            "He described the brake button as pressed equals HIGH with the button tied to 3V3."
        ),
    },
    {
        "questions": [
            "What pin mapping did he mention with the brake button?",
            "Which pins were used in his snippet?",
        ],
        "answer": (
            "In one snippet he listed encoder A/B on D2 and D3, with brake on D4."
        ),
    },
    {
        "questions": [
            "What force feedback symptoms did he report?",
            "How did he describe wheel feel problems?",
        ],
        "answer": (
            "He described soft force feedback, a false feeling of obstruction without mechanical blockage, and vibration-like behavior."
        ),
    },
    {
        "questions": [
            "Describe his sim rig constraints.",
            "How is he building the physical rig?",
        ],
        "answer": (
            "He is building a wood-only rig using only 2x4s, mostly with a drill as his tool, and discussed six 2x4x96 pieces plus a rear laptop stand."
        ),
    },
    {
        "questions": [
            "What extra parts does he want for the sim setup?",
            "Which CAD peripherals did he ask for?",
        ],
        "answer": (
            "He asked for a fist-shaped handbrake handle STEP file, limit-switch paddle shifter CAD files, and tensioner CAD parts."
        ),
    },
    {
        "questions": [
            "What wireless idea did he ask about?",
            "What board did he mention for wireless conversion?",
        ],
        "answer": "He asked whether the wheel could be made wireless using an ESP32.",
    },
    {
        "questions": [
            "What coding platform preference does he have?",
            "Does he prefer Raspberry Pi or Arduino examples?",
        ],
        "answer": "He prefers Raspberry Pi examples and generally prefers Python-based examples.",
    },
    {
        "questions": [
            "What CAD tools is he good with?",
            "Which CAD platforms did he mention confidence in?",
        ],
        "answer": "He said he is strong with SolidWorks and Onshape.",
    },
    {
        "questions": [
            "What common low-cost robotics parts does he use?",
            "List some hardware components associated with him.",
        ],
        "answer": (
            "He commonly referenced MG90S servos, HC-SR04 ultrasonic sensors, TB6612 drivers, and 28BYJ-48 steppers, with budget parts under $20."
        ),
    },
    {
        "questions": [
            "What higher-level opportunities has he aimed for?",
            "What academic networking goal did he mention?",
        ],
        "answer": "He has aimed to connect with an MIT professor.",
    },
    {
        "questions": [
            "What programs has he worked on applications for?",
            "Which application contexts did he mention?",
        ],
        "answer": "He mentioned SRA and ROAR Academy application work.",
    },
    {
        "questions": [
            "What writing preference does he have when editing drafts?",
            "How does he want rewrite requests handled?",
        ],
        "answer": "He prefers full rewrites instead of patch-style edits.",
    },
    {
        "questions": [
            "What response style does he prefer for technical work?",
            "How should robotics/code help be delivered?",
        ],
        "answer": (
            "For robotics and code, he likes clarifying questions when needed and clear, actionable outputs."
        ),
    },
    {
        "questions": [
            "What LinkedIn URL has he provided?",
            "Do we have his LinkedIn profile link?",
        ],
        "answer": "His provided LinkedIn URL is https://www.linkedin.com/in/pranav-emmadi-874723399/ .",
    },
]


UNKNOWN_QUESTIONS = [
    "What is his birthday?",
    "What is his home address?",
    "What is his phone number?",
    "What is his GPA?",
    "What college has accepted him?",
    "What are his bank details?",
    "What is his social security number?",
]


LONG_FORM_TASKS = [
    {
        "instruction": "Write a short About Me paragraph using only profile facts in 90 to 120 words.",
        "input": "Focus on robotics identity, sim wheel project, and leadership goals.",
        "output": (
            "Pranav Emmadi is a hands-on student builder from Evergreen, San Jose who enjoys fast iteration across robotics and hardware projects. He is active in FTC with Evergreen Dragons and joined FRC Team 2854 Prototypes with a mechanical focus. His recurring work includes a DIY robotic hand research effort, robotic arm exploration, FTC shooter iteration, and a custom sim racing wheel stack based on Arduino Leonardo, BTS7960, and quadrature encoder control. In sim work, he is actively debugging pedals, encoder scaling, and force-feedback behavior while testing a dual-motor summed-torque architecture. His leadership goals are mechanical lead by 10th grade and team captain by 11th grade."
        ),
    },
    {
        "instruction": "Create 6 resume-style bullets using only profile facts.",
        "input": "Keep each bullet concise and factual.",
        "output": (
            "- FTC member on Evergreen Dragons, focused on build-test-iterate hardware execution.\n"
            "- Joined FRC Team 2854 Prototypes with a mechanical-focused contribution path.\n"
            "- Built and tuned a DIY sim racing steering system using Arduino Leonardo and BTS7960.\n"
            "- Implemented quadrature encoder workflows using CPR = PPR x 4 and debugged real scaling errors.\n"
            "- Developed and documented a DIY robotic hand research direction: design, mechanics, and control.\n"
            "- Targets leadership growth: mechanical lead by 10th grade and team captain by 11th grade."
        ),
    },
    {
        "instruction": "Generate a concise technical summary of his sim wheel stack.",
        "input": "Mention hardware, control, and current in-progress upgrades.",
        "output": (
            "His sim wheel stack centers on an Arduino Leonardo with a BTS7960 driver and quadrature encoder feedback. He has validated the CPR = PPR x 4 rule, diagnosed angle-scaling mismatches, and debugged pedal axis behavior in Assetto Corsa. The current upgrade under test is a dual 12V brushed planetary gearmotor architecture geared into a single shaft to sum torque, with synchronized command behavior for both motors."
        ),
    },
]


def build_examples(seed: int = 3407) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []

    for item in QA_BANK:
        questions = item["questions"]
        answer = str(item["answer"])
        for question in questions:
            variants = [
                str(question),
                f"Profile check: {question}",
                f"In Pranav's saved profile, {str(question).lower()}",
            ]
            for variant in variants:
                examples.append(
                    {
                        "instruction": rng.choice(INSTRUCTION_POOL),
                        "input": variant,
                        "output": answer,
                    }
                )

    for question in UNKNOWN_QUESTIONS:
        examples.append(
            {
                "instruction": rng.choice(INSTRUCTION_POOL),
                "input": question,
                "output": UNKNOWN_OUTPUT,
            }
        )

    examples.extend(LONG_FORM_TASKS)
    rng.shuffle(examples)
    return examples


def write_jsonl(examples: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_and_write_dataset(output_path: Path, seed: int = 3407) -> int:
    examples = build_examples(seed=seed)
    write_jsonl(examples, output_path)
    return len(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Pranav profile dataset for LoRA fine-tuning."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pranav_profile_qa.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for example shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_and_write_dataset(args.output, seed=args.seed)
    print(f"Wrote {count} training examples to {args.output}")


if __name__ == "__main__":
    main()
