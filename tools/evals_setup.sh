#!/usr/bin/env bash

echo "Downloading text eval datasets (WikiText + Books)..."
echo "==================================================="

EVAL_DIR="text-evals/"
mkdir -p "$EVAL_DIR"

pip install -U -v datasets

echo
echo "Downloading WikiText-103 (test split)..."

python3 - <<EOF
from datasets import load_dataset
from pathlib import Path

out = Path("text-evals/ppl_wikitext_v1.txt")
ds = load_dataset("wikitext", "wikitext-103-v1", split="test")

with out.open("w") as f:
    for row in ds:
        text = row["text"].strip()
        if text and not text.startswith("="):
            f.write(text + "\n\n")

print(f"Wrote {out}")
EOF

echo
echo "Done."
echo "Files created:"
echo "  - text-evals/ppl_wikitext_v1.txt"
