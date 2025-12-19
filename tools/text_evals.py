#!/usr/bin/env python3
import os
import math
import time
from pathlib import Path
from tqdm import tqdm

from src.cactus_ffi import (
    cactus_init,
    cactus_destroy,
    cactus_tokenize,
    cactus_score_window,
)


MODEL_PATH = "../weights/lfm2-1.2b"
DATASET_PATH = "text-evals/ppl_wikitext_v1.txt"

MAX_DOCS   = 10
MAX_TOKENS = None

CTX_SIZE = 2048
STRIDE   = 512 

os.environ["CACTUS_KV_WINDOW_SIZE"] = str(CTX_SIZE)
os.environ["CACTUS_KV_SINK_SIZE"] = "0"

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

        print("Tokenizing...")
        tokens = cactus_tokenize(model, text)
        print(f"Tokenized: {len(tokens):,} tokens in {time.time() - t0:.2f}s")

        if MAX_TOKENS is not None and len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
            print(f"Capped to {len(tokens):,} tokens")

        total_logprob = 0.0
        total_tokens  = 0


        for i in tqdm(range(0, len(tokens), STRIDE), desc="Scoring PPL windows"):
            end = min(i + STRIDE, len(tokens))
            begin = max(end - CTX_SIZE, 0)

            window = tokens[begin:end]

            trg_len = end - i
            if trg_len <= 0:
                continue

            start_rel = len(window) - trg_len
            end_rel   = len(window)

            r = cactus_score_window(
                model,
                window,
                start_rel,
                end_rel,
                CTX_SIZE,
            )

            if not r.get("success", False):
                raise RuntimeError(r)

            total_logprob += r["logprob"]
            total_tokens  += r["tokens"]

        if total_tokens == 0:
            raise RuntimeError("No tokens were scored — this should never happen")

        avg_logprob = total_logprob / total_tokens
        ppl = math.exp(-avg_logprob)

        print("\nResults:")
        print(f"  Tokens scored: {total_tokens:,}")
        print(f"  Avg logprob:   {avg_logprob:.6f}")
        print(f"  Perplexity:   {ppl:.3f}")

    finally:
        cactus_destroy(model)


if __name__ == "__main__":
    main()
