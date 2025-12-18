#!/usr/bin/env python3
import math, time
from pathlib import Path
from tqdm import tqdm

from src.cactus_ffi import (
    cactus_init,
    cactus_destroy,
    cactus_tokenize,
    cactus_score_window,
)

MODEL_PATH = "../weights/lfm2-1.2b"
CTX_SIZE = 2048
MAX_DOCS = 50   # try 25 or 50


# Bigger stride = fewer windows = much faster
STRIDE = 2048   # try 2048 first; 512 is expensive

DATASET_PATH = "text-evals/ppl_wikitext_v1.txt"

# Optional: cap tokens for a smoke test
MAX_TOKENS = 50_000   # set None for full eval

def main():
    print("Loading model...")
    model = cactus_init(MODEL_PATH, context_size=CTX_SIZE)

    try:
        print("Loading dataset...")
        t0 = time.time()
        raw = Path(DATASET_PATH).read_text(encoding="utf-8", errors="ignore")
        docs = [d for d in raw.split("\n\n") if d.strip()]

        docs = docs[:MAX_DOCS]
        text = "\n\n".join(docs)

        print(f"Using {len(docs)} documents")
        print(f"Dataset chars: {len(text):,}")


        print("Tokenizing (this is before tqdm)...")
        tokens = cactus_tokenize(model, text)
        print(f"Tokenized: {len(tokens):,} tokens in {time.time() - t0:.2f}s")

        if MAX_TOKENS is not None and len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
            print(f"Capped to {len(tokens):,} tokens for quick run")

        total_logprob = 0.0
        total_tokens = 0

        starts = range(1, len(tokens), STRIDE)
        for start in tqdm(starts, desc="Scoring PPL windows"):
            end = min(start + STRIDE, len(tokens))
            r = cactus_score_window(model, tokens, start, end, CTX_SIZE)
            if not r.get("success", False):
                continue
            total_logprob += r["logprob"]
            total_tokens  += r["tokens"]

        if total_tokens == 0:
            print("No tokens scored.")
            return

        avg = total_logprob / total_tokens
        ppl = math.exp(-avg)

        print("\nResults:")
        print(f"  Tokens scored:  {total_tokens}")
        print(f"  Avg logprob:    {avg:.6f}")
        print(f"  Perplexity:     {ppl:.3f}")

    finally:
        cactus_destroy(model)

if __name__ == "__main__":
    main()
