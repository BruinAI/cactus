import time
import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

AUDIO_FILE = "test.wav"
MODEL_NAME = "openai/whisper-medium"
WHISPER_SR = 16000
TOP_K = 10


def main():
    print("Loading Whisper FP32 model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # ----------------------------------------------------
    # Load audio → mel
    # ----------------------------------------------------
    audio, _ = librosa.load(AUDIO_FILE, sr=WHISPER_SR)
    mel = processor.feature_extractor(
        audio,
        sampling_rate=WHISPER_SR,
        return_tensors="pt",
    ).input_features  # [1, 80, T]

    # ----------------------------------------------------
    # Forced decoder prompt (same as your C++ code)
    # ----------------------------------------------------
    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    prompt_token_ids = [tok for _, tok in forced_ids]

    print("Forced prompt token IDs:", prompt_token_ids)
    print(
        "Forced prompt decoded:",
        tokenizer.decode(prompt_token_ids, skip_special_tokens=False),
    )

    # ====================================================
    # 1) RAW LOGITS (like your C++ graph: no HF logits processors)
    # ====================================================
    print("\n===== RAW FORWARD (NO LOGITS PROCESSORS) =====")

    with torch.no_grad():
        decoder_input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
        # This runs encoder + decoder and returns raw LM-head logits
        outputs = model(
            input_features=mel,
            decoder_input_ids=decoder_input_ids,
        )
        # logits: [batch=1, seq_len, vocab_size]
        logits_raw = outputs.logits

    # Take last time step (first *generated* step would attend to previous tokens,
    # but this is the logits for the last token in decoder_input_ids)
    step_logits_raw = logits_raw[:, -1, :]          # [1, vocab]
    scores_np_raw = step_logits_raw[0].cpu().numpy()  # [vocab]

    vocab_size = scores_np_raw.shape[0]
    print("Vocab size:", vocab_size)

    # Greedy argmax, like your C++ sampler
    raw_best_id = int(scores_np_raw.argmax())
    raw_best_text = tokenizer.decode([raw_best_id])

    print(f"Raw first-step argmax token id (C++-style): {raw_best_id}")
    print(f"Raw first-step argmax token text: {raw_best_text!r}")

    # Top-K for raw logits
    topk_idx_raw = scores_np_raw.argsort()[::-1][:TOP_K]
    print("\nTop-10 tokens by RAW logit:")
    for rank, tid in enumerate(topk_idx_raw, start=1):
        print(
            f"{rank:2d}. id={tid:5d}  logit={scores_np_raw[tid]:8.4f}  text={tokenizer.decode([int(tid)])!r}"
        )

    # ====================================================
    # 2) HF-PROCESSED LOGITS via generate() (for comparison)
    # ====================================================
    print("\n===== SINGLE-STEP GENERATE (HF-PROCESSED LOGITS) =====")

    with torch.no_grad():
        gen_out = model.generate(
            input_features=mel,
            forced_decoder_ids=forced_ids,
            max_new_tokens=1,            # only first step
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,          # post-processor scores
        )

    # outputs.scores is a list of length = #generated steps.
    # Each element is [batch, vocab_size] AFTER HF's logits processors.
    scores_step0 = gen_out.scores[0]            # [1, vocab]
    scores_np_proc = scores_step0[0].cpu().numpy()

    proc_best_id = int(scores_np_proc.argmax())
    proc_best_text = tokenizer.decode([proc_best_id])

    print(f"HF processed first-step token id: {proc_best_id}")
    print(f"HF processed first-step token text: {proc_best_text!r}")

    topk_idx_proc = scores_np_proc.argsort()[::-1][:TOP_K]
    print("\nTop-10 tokens by PROCESSED logit (HF generate):")
    for rank, tid in enumerate(topk_idx_proc, start=1):
        print(
            f"{rank:2d}. id={tid:5d}  logit={scores_np_proc[tid]:8.4f}  text={tokenizer.decode([int(tid)])!r}"
        )

    # ====================================================
    # 3) Optional: full transcription sanity check
    # ====================================================
    print("\n===== FULL HF GENERATE() CHECK =====")

    with torch.no_grad():
        gen_ids = model.generate(
            input_features=mel,
            forced_decoder_ids=forced_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
        )

    gen_ids_list = gen_ids[0].tolist()
    prompt_len = len(prompt_token_ids)
    gen_prompt = gen_ids_list[:prompt_len]
    gen_new = gen_ids_list[prompt_len:]

    print("All generated token IDs:", gen_ids_list)
    print("Prompt part:", gen_prompt)
    print("Newly generated token IDs:", gen_new)

    full_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    print("\nDecoded full sequence (no special stripping):")
    print(full_text)

    transcription = tokenizer.decode(gen_new, skip_special_tokens=True)
    print("\n===== FINAL TRANSCRIPTION (new tokens only) =====")
    print(transcription if transcription.strip() else "<EMPTY STRING>")
    print("=================================================")


if __name__ == "__main__":
    main()
