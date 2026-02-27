#!/usr/bin/env python
"""Ollama-first self-learning assistant with automatic background retraining."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from build_pranav_profile_dataset import build_and_write_dataset
except ModuleNotFoundError:
    from scripts.build_pranav_profile_dataset import build_and_write_dataset


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_model_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug[:48] if slug else "model"


def model_to_unsloth(model_name: str) -> Optional[str]:
    n = model_name.lower()
    if n.startswith("llama3.2:1b"):
        return "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    if n.startswith("llama3.2:3b") or n == "llama3.2":
        return "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    return None


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: int = 15) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._serve_proc: Optional[subprocess.Popen] = None

    def _request_json(self, path: str, payload: Optional[Dict] = None) -> Dict:
        url = f"{self.host}{path}"
        if payload is None:
            req = urllib.request.Request(url=url, method="GET")
        else:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url=url,
                method="POST",
                data=data,
                headers={"Content-Type": "application/json"},
            )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_running(self) -> bool:
        try:
            self._request_json("/api/tags")
            return True
        except Exception:  # noqa: BLE001
            return False

    def ensure_running(self, auto_start: bool = True) -> None:
        if self.is_running():
            return
        if not auto_start:
            raise RuntimeError(
                "Ollama is not running. Start it first (`ollama serve`) and retry."
            )
        print("System: Ollama is not running. Starting `ollama serve`...")
        self._serve_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        for _ in range(25):
            time.sleep(0.8)
            if self.is_running():
                print("System: Ollama server started.")
                return
        raise RuntimeError(
            "Failed to start Ollama automatically. Start it manually with `ollama serve`."
        )

    def list_models(self) -> List[str]:
        data = self._request_json("/api/tags")
        models = data.get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]

    def generate(
        self,
        model: str,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        data = self._request_json(
            "/api/generate",
            payload={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            },
        )
        return str(data.get("response", "")).strip()

    def create_model_from_modelfile(
        self,
        model_name: str,
        modelfile: Path,
        quantize: Optional[str] = None,
    ) -> bool:
        cmd = ["ollama", "create", model_name, "-f", str(modelfile)]
        if quantize:
            cmd += ["--quantize", quantize]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print("[AutoTrain] `ollama create` failed:\n" + proc.stdout)
            return False
        return True


@dataclass
class Interaction:
    uid: str
    ts: str
    user_input: str
    model_output: str
    memory_context: str
    quality_score: float
    approved: bool
    source_model: str


class MemoryStore:
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
                    source_model TEXT NOT NULL
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
                (id, ts, user_input, model_output, memory_context, quality_score, approved, source_model)
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
                    row.source_model,
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
                "UPDATE interactions SET approved = 0, quality_score = 0.0 WHERE id = ?",
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
        self, base_dataset: Path, output_dataset: Path, min_quality: float
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
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


class ModelState:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._version = 0
        self._lock = threading.Lock()

    def get(self) -> Tuple[str, int]:
        with self._lock:
            return self._model_name, self._version

    def set(self, model_name: str) -> None:
        with self._lock:
            self._model_name = model_name
            self._version += 1


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
    user_input: str, memory_rows: Sequence[sqlite3.Row], top_k: int = 5
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
    out: List[str] = []
    for idx, (_, row) in enumerate(best, start=1):
        out.append(f"- Memory {idx}: Q: {row['user_input']} | A: {row['model_output']}")
    return "\n".join(out)


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
    if len(user_input.strip()) < 3:
        score = min(score, 0.4)
    return max(0.0, min(1.0, score))


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
                    "source_model": row.source_model,
                    "memory_context": row.memory_context,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_eval_ollama(client: OllamaClient, model_name: str, min_pass: int = 2) -> bool:
    tests = [
        ("What FTC team is Pranav on?", ["evergreen dragons"]),
        ("What is his preferred name?", ["superman"]),
        ("What CAD software does he use?", ["solidworks", "onshape"]),
        ("What is his birthday?", ["do not have that detail"]),
    ]
    passed = 0
    for question, must_have in tests:
        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            memory_block="- No relevant prior memory found.",
            user_input=question,
        )
        try:
            out = client.generate(
                model=model_name,
                prompt=prompt,
                max_new_tokens=160,
                temperature=0.2,
                top_p=0.9,
            ).lower()
        except Exception:  # noqa: BLE001
            return False
        if all(m in out for m in must_have):
            passed += 1
    return passed >= min_pass


class AutoTrainer:
    def __init__(
        self,
        client: OllamaClient,
        model_state: ModelState,
        python_exe: str,
        train_script: Path,
        base_dataset: Path,
        auto_dataset: Path,
        runtime_dir: Path,
        outputs_root: Path,
        base_ollama_model: str,
        unsloth_model_name: str,
        setup_script: Path,
        quantize: str,
        eval_min_pass: int,
        min_quality: float,
        retrain_every: int,
        train_steps: int,
    ) -> None:
        self.client = client
        self.model_state = model_state
        self.python_exe = python_exe
        self.train_script = train_script
        self.base_dataset = base_dataset
        self.auto_dataset = auto_dataset
        self.runtime_dir = runtime_dir
        self.outputs_root = outputs_root
        self.base_ollama_model = base_ollama_model
        self.unsloth_model_name = unsloth_model_name
        self.setup_script = setup_script
        self.quantize = quantize
        self.eval_min_pass = eval_min_pass
        self.min_quality = min_quality
        self.retrain_every = retrain_every
        self.train_steps = train_steps

        self.stage_output = self.outputs_root / "ollama_auto_stage"
        slug = sanitize_model_slug(base_ollama_model)
        self.deploy_model_name = f"pranav-auto-{slug}"

        self.store: Optional[MemoryStore] = None
        self.last_trained_candidate_count = 0
        self.training_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pending_requests: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)

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
                f"(delta={delta}). Starting retrain..."
            )

            cmd = [
                self.python_exe,
                str(self.train_script),
                "--dataset-path",
                str(self.auto_dataset),
                "--output-dir",
                str(self.stage_output),
                "--model-name",
                self.unsloth_model_name,
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
                "--save-merged-16bit",
            ]

            env = os.environ.copy()
            env["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
            proc = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
            if proc.returncode != 0:
                print("[AutoTrain] Training failed. Keeping current model.")
                return

            stage_adapter = self.stage_output / "pranav_lora"
            if not stage_adapter.exists():
                print("[AutoTrain] Stage adapter missing after training.")
                return

            stage_merged = self.stage_output / "pranav_merged_16bit"
            if not stage_merged.exists():
                print("[AutoTrain] Stage merged model missing after training.")
                return
            if not self.setup_script.exists():
                print(f"[AutoTrain] Setup script not found: {self.setup_script}")
                return

            build_cmd = [
                self.python_exe,
                str(self.setup_script),
                "--input-dir",
                str(stage_merged),
                "--model-name",
                self.deploy_model_name,
                "--quantize",
                self.quantize,
                "--python-exe",
                self.python_exe,
            ]
            print(f"[AutoTrain] Building Ollama model `{self.deploy_model_name}` ...")
            proc = subprocess.run(
                build_cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                print("[AutoTrain] Ollama model build failed:\n" + proc.stdout)
                return

            print("[AutoTrain] Running eval gate...")
            if not run_eval_ollama(
                self.client, self.deploy_model_name, min_pass=self.eval_min_pass
            ):
                print("[AutoTrain] Eval gate failed. Keeping current model.")
                return

            self.model_state.set(self.deploy_model_name)
            self.last_trained_candidate_count = candidate_count
            print(
                f"[AutoTrain] Promotion complete. Live model switched to `{self.deploy_model_name}`."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama autopilot: model select + chat + auto-train + auto-rollout."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama API host.",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/pranav_profile_qa.jsonl"),
        help="Base dataset path.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime/ollama_autopilot"),
        help="Runtime directory for logs, DB, and Modelfiles.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=str(Path(".venv311/Scripts/python.exe"))
        if Path(".venv311/Scripts/python.exe").exists()
        else sys.executable,
        help="Python executable used for background retraining.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/train_pranav_8gb.py"),
        help="Retraining script path.",
    )
    parser.add_argument(
        "--setup-script",
        type=Path,
        default=Path("scripts/setup_pranav_ollama.py"),
        help="Script used to convert merged model and register in Ollama.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.90,
        help="Auto-promotion quality threshold.",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=40,
        help="Retrain after this many new promoted interactions.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Background retraining steps per cycle.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="q4_K_M",
        help="Quantization used for auto-created Ollama rollouts.",
    )
    parser.add_argument(
        "--eval-min-pass",
        type=int,
        default=2,
        help="Minimum number of evaluation checks required to promote a retrained model.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=5,
        help="Retrieved memory snippets injected per prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Generation max new tokens.",
    )
    parser.add_argument(
        "--auto-start-ollama",
        action="store_true",
        help="Auto-start Ollama server if not running.",
    )
    return parser.parse_args()


def choose_supported_model(client: OllamaClient) -> str:
    models = client.list_models()
    supported = [m for m in models if model_to_unsloth(m) is not None]
    if not supported:
        raise RuntimeError(
            "No supported Ollama model found.\n"
            "Install one with: ollama pull llama3.2:1b"
        )
    print("\nSelect an Ollama model for autopilot training:")
    for idx, name in enumerate(supported, start=1):
        print(f"  {idx}. {name}")
    while True:
        pick = input("Model number: ").strip()
        if pick.isdigit():
            i = int(pick)
            if 1 <= i <= len(supported):
                return supported[i - 1]
        print("Invalid selection. Try again.")


def ensure_base_dataset(path: Path) -> None:
    if path.exists():
        return
    print(f"System: Base dataset missing at {path}. Building default dataset...")
    count = build_and_write_dataset(path)
    print(f"System: Built base dataset with {count} rows.")


def main() -> None:
    args = parse_args()
    ensure_base_dataset(args.base_dataset)

    client = OllamaClient(host=args.host, timeout=30)
    client.ensure_running(auto_start=args.auto_start_ollama)

    base_model = choose_supported_model(client)
    unsloth_model = model_to_unsloth(base_model)
    if unsloth_model is None:
        raise RuntimeError("Selected model is unsupported for automatic retraining.")

    model_state = ModelState(base_model)
    store = MemoryStore(args.runtime_dir / "memory.db")
    interaction_log = args.runtime_dir / "interactions.jsonl"
    auto_dataset = args.runtime_dir / "auto_training_dataset.jsonl"

    trainer = AutoTrainer(
        client=client,
        model_state=model_state,
        python_exe=args.python_exe,
        train_script=args.train_script,
        base_dataset=args.base_dataset,
        auto_dataset=auto_dataset,
        runtime_dir=args.runtime_dir,
        outputs_root=Path("outputs"),
        base_ollama_model=base_model,
        unsloth_model_name=unsloth_model,
        setup_script=args.setup_script,
        quantize=args.quantize,
        eval_min_pass=args.eval_min_pass,
        min_quality=args.min_quality,
        retrain_every=args.retrain_every,
        train_steps=args.train_steps,
    )
    trainer.attach_store(store)
    trainer.start()

    last_interaction_id: Optional[str] = None
    last_seen_model, last_seen_version = model_state.get()
    print(
        "\nOllama Autopilot ready.\n"
        f"Live model: {last_seen_model}\n"
        "Commands:\n"
        "  /approve-last  -> force-approve last response for training\n"
        "  /bad-last      -> mark last response bad\n"
        "  /stats         -> show counts\n"
        "  /model         -> show current live model\n"
        "  exit           -> quit\n"
    )

    try:
        while True:
            live_model, live_version = model_state.get()
            if live_version != last_seen_version:
                print(f"System: Switched live model to `{live_model}`.")
                last_seen_model = live_model
                last_seen_version = live_version

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
                    print("System: No interaction to approve.")
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
            if user_input == "/model":
                print(f"System: live_model={live_model}")
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

            try:
                reply = client.generate(
                    model=live_model,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.3,
                    top_p=0.9,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"System: generation failed: {exc}")
                continue

            if not reply.strip():
                reply = "I do not have that detail in Pranav's saved profile yet."
            print(f"Model: {reply}")

            q_score = quality_heuristic(user_input=user_input, model_output=reply)
            approved = q_score >= args.min_quality
            interaction = Interaction(
                uid=str(uuid.uuid4()),
                ts=now_iso(),
                user_input=user_input,
                model_output=reply,
                memory_context=memory_context,
                quality_score=q_score,
                approved=approved,
                source_model=live_model,
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
