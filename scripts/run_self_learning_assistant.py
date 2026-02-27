#!/usr/bin/env python
"""Run PranavProfileGPT with automatic memory + logging + retraining loop."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are PranavProfileGPT, a profile-grounded assistant.\n"
    "Rules:\n"
    "- Use only known profile facts and retrieved memory context.\n"
    "- If unknown, say you do not have that detail yet.\n"
    "- Do not invent personal details."
)


PROMPT_TEMPLATE = """{system_prompt}

### Retrieved Memory:
{memory_block}

### Instruction:
Answer the user question about Pranav's profile using facts only.

### Input:
{user_input}

### Response:
"""


TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class Interaction:
    uid: str
    ts: str
    user_input: str
    model_output: str
    memory_context: str
    quality_score: float
    approved: bool
    source_adapter: str


class MemoryStore:
    """Simple SQLite memory store with token-overlap retrieval."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    memory_context TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    approved INTEGER NOT NULL,
                    source_adapter TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS train_promoted (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def add_interaction(self, row: Interaction) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interactions
                (id, ts, user_input, model_output, memory_context, quality_score, approved, source_adapter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.uid,
                    row.ts,
                    row.user_input,
                    row.model_output,
                    row.memory_context,
                    row.quality_score,
                    1 if row.approved else 0,
                    row.source_adapter,
                ),
            )
            conn.commit()

    def promote_interaction(self, uid: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, ts, user_input, model_output
                FROM interactions
                WHERE id = ?
                """,
                (uid,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE interactions
                SET approved = 1, quality_score = CASE WHEN quality_score < 0.95 THEN 0.95 ELSE quality_score END
                WHERE id = ?
                """,
                (uid,),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO train_promoted (id, ts, instruction, input, output)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["ts"],
                    "Answer the profile question from Pranav's known facts.",
                    row["user_input"],
                    row["model_output"],
                ),
            )
            conn.commit()
            return True

    def mark_last_bad(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM interactions ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE interactions
                SET approved = 0, quality_score = 0.0
                WHERE id = ?
                """,
                (row["id"],),
            )
            conn.execute("DELETE FROM train_promoted WHERE id = ?", (row["id"],))
            conn.commit()
            return True

    def candidate_count(self, min_quality: float) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM interactions
                WHERE approved = 1 AND quality_score >= ?
                """,
                (min_quality,),
            ).fetchone()
            return int(row["c"])

    def fetch_recent(self, limit: int = 2000) -> List[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ts, user_input, model_output, quality_score, approved
                FROM interactions
                ORDER BY ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return list(rows)

    def build_auto_dataset(
        self,
        base_dataset: Path,
        output_dataset: Path,
        min_quality: float,
    ) -> int:
        output_dataset.parent.mkdir(parents=True, exist_ok=True)
        merged: List[Dict[str, str]] = []

        with base_dataset.open("r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ts, user_input, model_output
                FROM interactions
                WHERE approved = 1 AND quality_score >= ?
                ORDER BY ts ASC
                """,
                (min_quality,),
            ).fetchall()

        for row in rows:
            merged.append(
                {
                    "instruction": "Answer the profile question from Pranav's known facts.",
                    "input": row["user_input"],
                    "output": row["model_output"],
                }
            )

        with output_dataset.open("w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return len(rows)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def retrieval_score(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    q = set(query_tokens)
    c = set(candidate_tokens)
    overlap = len(q.intersection(c))
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(q))


def select_memory_context(
    user_input: str,
    memory_rows: Sequence[sqlite3.Row],
    top_k: int = 5,
) -> str:
    q_tokens = tokenize(user_input)
    scored: List[Tuple[float, sqlite3.Row]] = []
    for row in memory_rows:
        text = f"{row['user_input']} {row['model_output']}"
        score = retrieval_score(q_tokens, tokenize(text))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[:top_k]
    if not best:
        return "- No relevant prior memory found."
    lines = []
    for idx, (_, row) in enumerate(best, start=1):
        lines.append(
            f"- Memory {idx}: Q: {row['user_input']} | A: {row['model_output']}"
        )
    return "\n".join(lines)


def quality_heuristic(user_input: str, model_output: str) -> float:
    text = model_output.strip().lower()
    if not text:
        return 0.0
    if "i do not have that detail" in text:
        return 0.95
    score = 0.6
    if len(model_output) >= 25:
        score += 0.15
    if len(model_output) <= 700:
        score += 0.1
    if any(
        kw in text
        for kw in [
            "evergreen dragons",
            "pranav",
            "frc",
            "ftc",
            "arduino leonardo",
            "bts7960",
            "solidworks",
            "onshape",
            "raspberry pi",
        ]
    ):
        score += 0.15
    if "http" in text and "linkedin.com/in/pranav-emmadi-874723399" not in text:
        score -= 0.2
    if len(user_input.strip()) < 3:
        score = min(score, 0.4)
    return max(0.0, min(1.0, score))


def run_eval(adapter_dir: Path) -> bool:
    # Lightweight keyword-based smoke test before rollout.
    def check_terms(text: str, terms: Sequence[str]) -> bool:
        low = text.lower()
        return all(term in low for term in terms)

    def check_bts7960(text: str) -> bool:
        low = text.lower()
        if "bts7960" not in low:
            return False
        return not any(x in low for x in ["bts73", "bts78", "bts 73", "bts 78"])

    def check_cpr_formula(text: str) -> bool:
        low = text.lower()
        if "cpr" not in low or "ppr" not in low:
            return False
        has_x4 = any(token in low for token in ["x 4", "x4", "* 4", "*4"])
        if not has_x4:
            return False
        return not any(x in low for x in ["4/3", "x 4/3", "x4/3", "*4/3"])

    tests = [
        ("What FTC team is Pranav on?", lambda t: check_terms(t, ["evergreen dragons"])),
        ("What is his preferred name?", lambda t: check_terms(t, ["superman"])),
        ("What CAD software does he use?", lambda t: check_terms(t, ["solidworks", "onshape"])),
        ("What motor driver does his wheel use?", check_bts7960),
        ("How does he define CPR from PPR?", check_cpr_formula),
        ("What is his birthday?", lambda t: check_terms(t, ["do not have that detail"])),
    ]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    passes = 0
    for question, validator in tests:
        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            memory_block="- No relevant prior memory found.",
            user_input=question,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
        )
        text = tokenizer.batch_decode(output)[0].lower()
        if validator(text):
            passes += 1

    return passes >= 5


class SelfTrainer:
    """Background retraining loop."""

    def __init__(
        self,
        python_exe: str,
        train_script: Path,
        base_dataset: Path,
        auto_dataset: Path,
        outputs_root: Path,
        runtime_dir: Path,
        min_quality: float,
        retrain_every: int,
        train_steps: int,
    ) -> None:
        self.python_exe = python_exe
        self.train_script = train_script
        self.base_dataset = base_dataset
        self.auto_dataset = auto_dataset
        self.outputs_root = outputs_root
        self.stage_output = outputs_root / "auto_stage"
        self.prod_output = outputs_root / "pranav_8gb"
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.min_quality = min_quality
        self.retrain_every = retrain_every
        self.train_steps = train_steps
        self.training_lock = threading.Lock()
        self.pending_requests = queue.Queue()
        self.last_trained_candidate_count = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.store: MemoryStore | None = None
        self._promotion_version = 0

    def attach_store(self, store: MemoryStore) -> None:
        self.store = store

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pending_requests.put("stop")
        self.thread.join(timeout=5)

    def notify_new_data(self) -> None:
        self.pending_requests.put("new_data")

    @property
    def promotion_version(self) -> int:
        return self._promotion_version

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                _ = self.pending_requests.get(timeout=2)
            except queue.Empty:
                continue
            if self.stop_event.is_set():
                return
            self._maybe_retrain()

    def _maybe_retrain(self) -> None:
        if self.store is None:
            return
        with self.training_lock:
            candidate_count = self.store.candidate_count(self.min_quality)
            delta = candidate_count - self.last_trained_candidate_count
            if delta < self.retrain_every:
                return

            added = self.store.build_auto_dataset(
                base_dataset=self.base_dataset,
                output_dataset=self.auto_dataset,
                min_quality=self.min_quality,
            )
            print(
                f"\n[AutoTrain] Triggered with {added} promoted interactions "
                f"(delta={delta}). Starting background training..."
            )

            cmd = [
                self.python_exe,
                str(self.train_script),
                "--dataset-path",
                str(self.auto_dataset),
                "--output-dir",
                str(self.stage_output),
                "--model-name",
                "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                "--max-seq-length",
                "1024",
                "--batch-size",
                "1",
                "--grad-accum",
                "8",
                "--max-steps",
                str(self.train_steps),
                "--learning-rate",
                "2e-4",
            ]

            env = os.environ.copy()
            env["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
            proc = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
            if proc.returncode != 0:
                print("[AutoTrain] Training failed. Keeping current production model.")
                return

            stage_adapter = self.stage_output / "pranav_lora"
            prod_adapter = self.prod_output / "pranav_lora"
            if not stage_adapter.exists():
                print("[AutoTrain] Stage adapter missing after training.")
                return

            print("[AutoTrain] Running post-train evaluation...")
            try:
                passed = run_eval(stage_adapter)
            except Exception as exc:  # noqa: BLE001
                print(f"[AutoTrain] Eval failed with error: {exc}")
                passed = False

            if not passed:
                print("[AutoTrain] Eval gate failed. Model not promoted.")
                return

            backup_root = self.runtime_dir / "backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / f"pranav_lora_{int(time.time())}"
            if prod_adapter.exists():
                shutil.copytree(prod_adapter, backup_dir, dirs_exist_ok=True)

            prod_adapter.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(prod_adapter, ignore_errors=True)
            shutil.copytree(stage_adapter, prod_adapter)

            self.last_trained_candidate_count = candidate_count
            self._promotion_version += 1
            print("[AutoTrain] Promotion complete. New model is now live.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run self-learning Pranav profile assistant."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/pranav_8gb/pranav_lora"),
        help="Current production adapter dir.",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/pranav_profile_qa.jsonl"),
        help="Base profile dataset path.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime/self_learning"),
        help="Runtime directory for DB/logs/backups.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used for auto-retraining subprocess.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/train_pranav_8gb.py"),
        help="Training script path for background retraining.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.85,
        help="Minimum quality score for auto-promotion to training candidates.",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=25,
        help="Trigger retrain after this many newly promoted interactions.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=120,
        help="Background auto-train steps per cycle.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=5,
        help="How many memories to inject into prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Generation max new tokens.",
    )
    return parser.parse_args()


def write_interaction_jsonl(log_path: Path, row: Interaction) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": row.uid,
                    "ts": row.ts,
                    "input": row.user_input,
                    "output": row.model_output,
                    "quality_score": row.quality_score,
                    "approved": row.approved,
                    "source_adapter": row.source_adapter,
                    "memory_context": row.memory_context,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def load_model(adapter_dir: Path):
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_dir}. Train once first."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if not args.base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {args.base_dataset}")

    db_path = args.runtime_dir / "memory.db"
    interaction_log = args.runtime_dir / "interactions.jsonl"
    auto_dataset = args.runtime_dir / "auto_training_dataset.jsonl"

    store = MemoryStore(db_path=db_path)
    trainer = SelfTrainer(
        python_exe=args.python_exe,
        train_script=args.train_script,
        base_dataset=args.base_dataset,
        auto_dataset=auto_dataset,
        outputs_root=Path("outputs"),
        runtime_dir=args.runtime_dir,
        min_quality=args.min_quality,
        retrain_every=args.retrain_every,
        train_steps=args.train_steps,
    )
    trainer.attach_store(store)
    trainer.start()

    print("Loading production adapter...")
    model, tokenizer = load_model(args.adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seen_promotion_version = trainer.promotion_version

    last_interaction_id: str | None = None
    print(
        "\nSelf-learning PranavProfileGPT ready.\n"
        "Commands:\n"
        "  /approve-last  -> force-approve last response for training\n"
        "  /bad-last      -> mark last response bad and exclude from training\n"
        "  /stats         -> show memory and candidate stats\n"
        "  /reload        -> reload current production adapter\n"
        "  exit           -> quit\n"
    )

    try:
        while True:
            if trainer.promotion_version != seen_promotion_version:
                model, tokenizer = load_model(args.adapter_dir)
                seen_promotion_version = trainer.promotion_version
                print("System: Auto-reloaded newly promoted model.")

            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input == "/approve-last":
                if last_interaction_id and store.promote_interaction(last_interaction_id):
                    print("System: Last interaction approved and promoted.")
                    trainer.notify_new_data()
                else:
                    print("System: No interaction available to approve.")
                continue
            if user_input == "/bad-last":
                ok = store.mark_last_bad()
                print("System: Marked last response bad." if ok else "System: Nothing to mark.")
                continue
            if user_input == "/stats":
                total = len(store.fetch_recent(limit=100000))
                candidates = store.candidate_count(args.min_quality)
                print(
                    f"System: total_interactions={total}, "
                    f"promoted_candidates={candidates}, "
                    f"retrain_every={args.retrain_every}"
                )
                continue
            if user_input == "/reload":
                model, tokenizer = load_model(args.adapter_dir)
                print("System: Reloaded production adapter.")
                continue

            memory_rows = store.fetch_recent(limit=2000)
            memory_context = select_memory_context(
                user_input=user_input,
                memory_rows=memory_rows,
                top_k=args.memory_top_k,
            )

            prompt = PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                memory_block=memory_context,
                user_input=user_input,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                use_cache=True,
            )
            full_text = tokenizer.batch_decode(output_tokens)[0]
            reply = full_text.split("### Response:")[-1].strip()
            if "<|eot_id|>" in reply:
                reply = reply.split("<|eot_id|>")[0].strip()
            if not reply:
                reply = "I do not have that detail in Pranav's saved profile yet."

            print(f"Model: {reply}")

            q_score = quality_heuristic(user_input=user_input, model_output=reply)
            approved = q_score >= args.min_quality
            interaction = Interaction(
                uid=str(uuid.uuid4()),
                ts=datetime.now(timezone.utc).isoformat(),
                user_input=user_input,
                model_output=reply,
                memory_context=memory_context,
                quality_score=q_score,
                approved=approved,
                source_adapter=str(args.adapter_dir),
            )
            store.add_interaction(interaction)
            write_interaction_jsonl(interaction_log, interaction)
            last_interaction_id = interaction.uid
            if approved:
                store.promote_interaction(interaction.uid)
                trainer.notify_new_data()
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()
