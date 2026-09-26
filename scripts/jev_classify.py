#!/usr/bin/env python
"""Classify many texts against one Kalshi market rule with TypeSafe Jev.

Jev (~typesafe/jev-latest, OpenRouter Decisions API) is fast and accurate at
"does this text satisfy this rule?" — word forms, speaker, what counts — and
bad at arithmetic. Use it as a first-pass filter over transcripts/posts/
headlines, then confirm the hits yourself.

Usage:
    python scripts/jev_classify.py input.json > out.json
    input.json = {"rule": "<exact market rule>", "context": {...optional...},
                  "items": [{"id": "...", "text": "..."}, ...]}
Output: [{"id", "p_true"}] sorted by p_true desc (null if a batch failed).
Needs OPENROUTER_API_KEY in the environment or .env.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.agent.jev import classify_texts  # noqa: E402


def main() -> None:
    load_dotenv()
    inp = json.loads(Path(sys.argv[1]).read_text())
    items = inp["items"]
    probs = asyncio.run(classify_texts(inp["rule"], [it["text"] for it in items], inp.get("context")))
    out = [{"id": it["id"], "p_true": p} for it, p in zip(items, probs)]
    print(json.dumps(sorted(out, key=lambda x: -(x["p_true"] or 0)), indent=1))


if __name__ == "__main__":
    main()
