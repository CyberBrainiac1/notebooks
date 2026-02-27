#!/usr/bin/env python
"""Ollama assistant with vector memory (no retraining required)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
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


UNKNOWN_REPLY = "I do not have that detail in Pranav's saved profile yet."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


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


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: int = 30) -> None:
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

    def pull_model(self, model_name: str) -> None:
        cmd = ["ollama", "pull", model_name]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to pull Ollama model: {model_name}")

    def generate(
        self,
        model: str,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
        top_p: float = 0.85,
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

    def embed_texts(self, model: str, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # New endpoint.
        try:
            data = self._request_json(
                "/api/embed",
                payload={"model": model, "input": list(texts)},
            )
            embs = data.get("embeddings")
            if isinstance(embs, list) and len(embs) == len(texts):
                return [list(map(float, e)) for e in embs]
        except Exception:  # noqa: BLE001
            pass

        # Fallback endpoint.
        out: List[List[float]] = []
        for text in texts:
            data = self._request_json(
                "/api/embeddings",
                payload={"model": model, "prompt": text},
            )
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Ollama embedding response missing `embedding`.")
            out.append(list(map(float, emb)))
        return out


@dataclass
class Interaction:
    uid: str
    ts: str
    user_input: str
    model_output: str
    memory_context: str
    quality_score: float
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
                    source_model TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(ts DESC);"
            )
            conn.commit()

    @staticmethod
    def _memory_id(content: str) -> str:
        norm = content.strip().lower()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def add_memory(
        self,
        content: str,
        embedding: Sequence[float],
        source: str,
        ts: Optional[str] = None,
    ) -> bool:
        memory_id = self._memory_id(content)
        ts = ts or now_iso()
        emb_json = json.dumps(list(embedding), separators=(",", ":"))
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO memories (id, ts, source, content, embedding_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, ts, source, content, emb_json),
            )
            conn.commit()
            return cur.rowcount > 0

    def add_interaction(self, row: Interaction) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interactions
                (id, ts, user_input, model_output, memory_context, quality_score, source_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.uid,
                    row.ts,
                    row.user_input,
                    row.model_output,
                    row.memory_context,
                    row.quality_score,
                    row.source_model,
                ),
            )
            conn.commit()

    def memory_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
            return int(row["c"])

    def interaction_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM interactions").fetchone()
            return int(row["c"])

    def fetch_memory_rows(self, limit: int = 5000) -> List[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ts, source, content, embedding_json
                FROM memories
                ORDER BY ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return list(rows)

    def seed_from_dataset(
        self,
        dataset_path: Path,
        embed_fn,
        batch_size: int = 24,
    ) -> int:
        if not dataset_path.exists():
            return 0
        new_contents: List[str] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = str(row.get("input", "")).strip()
                a = str(row.get("output", "")).strip()
                if not q or not a:
                    continue
                content = f"Known memory: Q: {q} | A: {a}"
                new_contents.append(content)

        # Filter content that already exists.
        pending: List[str] = []
        with self._connect() as conn:
            for content in new_contents:
                mid = self._memory_id(content)
                row = conn.execute(
                    "SELECT 1 FROM memories WHERE id = ? LIMIT 1", (mid,)
                ).fetchone()
                if row is None:
                    pending.append(content)

        added = 0
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            embeddings = embed_fn(batch)
            for content, emb in zip(batch, embeddings):
                if self.add_memory(content=content, embedding=emb, source="dataset"):
                    added += 1
        return added

    def retrieve_by_embedding(
        self,
        query_embedding: Sequence[float],
        top_k: int = 6,
        scan_limit: int = 5000,
    ) -> List[Tuple[float, sqlite3.Row]]:
        rows = self.fetch_memory_rows(limit=scan_limit)
        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            try:
                emb = json.loads(row["embedding_json"])
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(emb, list):
                continue
            score = cosine_similarity(query_embedding, emb)
            if score <= 0:
                continue
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def choose_chat_model(client: OllamaClient) -> str:
    models = client.list_models()
    # Exclude typical embedding-only models from chat choices.
    chat_models = [
        m for m in models if "embed" not in m.lower() and "embedding" not in m.lower()
    ]
    if not chat_models:
        raise RuntimeError(
            "No chat model found in Ollama. Install one with: ollama pull llama3.2:3b"
        )
    print("\nSelect an Ollama chat model:")
    for idx, name in enumerate(chat_models, start=1):
        print(f"  {idx}. {name}")
    while True:
        pick = input("Model number: ").strip()
        if pick.isdigit():
            i = int(pick)
            if 1 <= i <= len(chat_models):
                return chat_models[i - 1]
        print("Invalid selection. Try again.")


def ensure_base_dataset(path: Path) -> None:
    if path.exists():
        return
    print(f"System: Base dataset missing at {path}. Building default dataset...")
    count = build_and_write_dataset(path)
    print(f"System: Built base dataset with {count} rows.")


def format_memory_context(
    store: MemoryStore,
    client: OllamaClient,
    embed_model: str,
    user_input: str,
    top_k: int,
    scan_limit: int,
) -> str:
    query_emb = client.embed_texts(embed_model, [user_input])[0]
    hits = store.retrieve_by_embedding(
        query_embedding=query_emb,
        top_k=top_k,
        scan_limit=scan_limit,
    )
    if not hits:
        return "- No relevant prior memory found."
    lines: List[str] = []
    for idx, (score, row) in enumerate(hits, start=1):
        lines.append(
            f"- Memory {idx} (score={score:.3f}, source={row['source']}): {row['content']}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama assistant with vector memory and no retraining."
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument(
        "--chat-model",
        type=str,
        default="llama3.1:8b",
        help="Chat model to use. Set empty string to pick interactively.",
    )
    parser.add_argument(
        "--auto-pull-chat-model",
        action="store_true",
        help="Auto-pull chat model if missing.",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/pranav_profile_qa_v4.jsonl"),
        help="Dataset used to seed initial memory facts.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime/ollama_vector_memory"),
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model used for vector memory retrieval.",
    )
    parser.add_argument(
        "--auto-pull-embed-model",
        action="store_true",
        help="Auto-pull embedding model if missing.",
    )
    parser.add_argument("--memory-top-k", type=int, default=6)
    parser.add_argument("--memory-scan-limit", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.9,
        help="Only interactions above this score are auto-stored as new memory.",
    )
    parser.add_argument("--auto-start-ollama", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_base_dataset(args.base_dataset)

    client = OllamaClient(host=args.host, timeout=40)
    client.ensure_running(auto_start=args.auto_start_ollama)

    models = client.list_models()
    if args.embed_model not in models and args.auto_pull_embed_model:
        print(f"System: pulling embedding model `{args.embed_model}` ...")
        client.pull_model(args.embed_model)
        models = client.list_models()

    if args.embed_model not in models:
        raise RuntimeError(
            f"Embedding model `{args.embed_model}` not found.\n"
            f"Run: ollama pull {args.embed_model}"
        )

    chat_model = args.chat_model.strip()
    if chat_model:
        if chat_model not in models and args.auto_pull_chat_model:
            print(f"System: pulling chat model `{chat_model}` ...")
            client.pull_model(chat_model)
            models = client.list_models()
        if chat_model not in models:
            raise RuntimeError(
                f"Chat model `{chat_model}` not found.\n"
                f"Run: ollama pull {chat_model}"
            )
    else:
        chat_model = choose_chat_model(client)
    store = MemoryStore(args.runtime_dir / "memory.db")
    interaction_log = args.runtime_dir / "interactions.jsonl"
    interaction_log.parent.mkdir(parents=True, exist_ok=True)

    print("System: seeding vector memory from base dataset...")
    seeded = store.seed_from_dataset(
        dataset_path=args.base_dataset,
        embed_fn=lambda texts: client.embed_texts(args.embed_model, texts),
    )
    print(
        f"System: seed complete (added={seeded}, total_memories={store.memory_count()})."
    )

    print(
        "\nVector Memory Assistant ready.\n"
        f"Live model: {chat_model}\n"
        f"Embedding model: {args.embed_model}\n"
        "Commands:\n"
        "  /remember <fact>  -> store a new fact directly in vector memory\n"
        "  /stats            -> show counts\n"
        "  /model            -> show current model\n"
        "  /help             -> show commands\n"
        "  exit              -> quit\n"
    )

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.startswith("/remember "):
            fact = user_input[len("/remember ") :].strip()
            if not fact:
                print("System: usage -> /remember <fact>")
                continue
            emb = client.embed_texts(args.embed_model, [fact])[0]
            added = store.add_memory(content=fact, embedding=emb, source="manual")
            print("System: memory saved." if added else "System: memory already exists.")
            continue
        if user_input == "/stats":
            print(
                f"System: interactions={store.interaction_count()}, "
                f"memories={store.memory_count()}, top_k={args.memory_top_k}"
            )
            continue
        if user_input == "/model":
            print(f"System: chat_model={chat_model}, embed_model={args.embed_model}")
            continue
        if user_input == "/help":
            print(
                "Commands: /remember <fact>, /stats, /model, /help, exit"
            )
            continue

        try:
            memory_context = format_memory_context(
                store=store,
                client=client,
                embed_model=args.embed_model,
                user_input=user_input,
                top_k=args.memory_top_k,
                scan_limit=args.memory_scan_limit,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"System: memory retrieval failed: {exc}")
            memory_context = "- No relevant prior memory found."

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            memory_block=memory_context,
            user_input=user_input,
        )

        try:
            reply = client.generate(
                model=chat_model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"System: generation failed: {exc}")
            continue

        if not reply.strip():
            reply = UNKNOWN_REPLY
        print(f"Model: {reply}")

        q_score = quality_heuristic(user_input=user_input, model_output=reply)
        interaction = Interaction(
            uid=str(uuid.uuid4()),
            ts=now_iso(),
            user_input=user_input,
            model_output=reply,
            memory_context=memory_context,
            quality_score=q_score,
            source_model=chat_model,
        )
        store.add_interaction(interaction)
        with interaction_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": interaction.uid,
                        "ts": interaction.ts,
                        "input": interaction.user_input,
                        "output": interaction.model_output,
                        "memory_context": interaction.memory_context,
                        "quality_score": interaction.quality_score,
                        "source_model": interaction.source_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if q_score >= args.min_quality:
            memory_text = f"Q: {user_input} | A: {reply}"
            try:
                emb = client.embed_texts(args.embed_model, [memory_text])[0]
                store.add_memory(
                    content=memory_text,
                    embedding=emb,
                    source="interaction",
                    ts=interaction.ts,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"System: failed to store interaction memory: {exc}")


if __name__ == "__main__":
    main()
