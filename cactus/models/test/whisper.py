import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

AUDIO_FILE = "test.wav"
MODEL_NAME = "openai/whisper-medium"
OUT_DIR = "./"
WHISPER_SR = 16000


def save(path, arr):
    np.save(path, arr)
    print(f"✅ Saved {path}  shape={tuple(arr.shape)}  dtype={arr.dtype}")


def main():
    print("Loading Whisper FP32 model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    # ----------------------------------------------------
    # Load audio → mel
    # ----------------------------------------------------
    audio, _ = librosa.load(AUDIO_FILE, sr=WHISPER_SR)

    mel = processor.feature_extractor(
        audio,
        sampling_rate=WHISPER_SR,
        return_tensors="pt"
    ).input_features.float()

    mel_np = mel.detach().cpu().numpy()
    save(f"{OUT_DIR}/mel.npy", mel_np)

    # ----------------------------------------------------
    # Conv1 + Conv2
    # ----------------------------------------------------
    with torch.no_grad():
        conv1 = model.model.encoder.conv1(mel)
        conv2 = model.model.encoder.conv2(conv1)

    save(f"{OUT_DIR}/conv1_out.npy", conv1.cpu().numpy())
    save(f"{OUT_DIR}/conv2_out.npy", conv2.cpu().numpy())

    conv2_T = conv2.transpose(1, 2)   # (1, T_enc, 1024)
    x = conv2_T

    T_enc = x.shape[1]
    D = x.shape[2]

    # ----------------------------------------------------
    # Positional embeddings + pre-LN
    # ----------------------------------------------------
    pos_emb = model.model.encoder.embed_positions.weight[:T_enc].detach()[None, :, :]
    save(f"{OUT_DIR}/encoder_pos_emb.npy", pos_emb[0].cpu().numpy())

    x = x + pos_emb
    save(f"{OUT_DIR}/encoder_pos_added.npy", x[0].cpu().numpy())

    w = model.model.encoder.layer_norm.weight.detach().cpu().numpy()
    b = model.model.encoder.layer_norm.bias.detach().cpu().numpy()

    enc_plus_pos = x[0].cpu().numpy()
    mean = enc_plus_pos.mean(-1, keepdims=True)
    var = enc_plus_pos.var(-1, keepdims=True)
    ln_np = (enc_plus_pos - mean) / np.sqrt(var + 1e-5)
    ln_np = ln_np * w + b
    save(f"{OUT_DIR}/encoder_post_norm.npy", ln_np)

    x = torch.tensor(ln_np, dtype=torch.float32).unsqueeze(0)

    # ----------------------------------------------------
    # Full encoder stack (manual)
    # ----------------------------------------------------
    for i, enc_layer in enumerate(model.model.encoder.layers):
        with torch.no_grad():
            ln1 = enc_layer.self_attn_layer_norm(x)
            sa_out = enc_layer.self_attn(ln1)[0]
            x_sa = x + sa_out

            ln2 = enc_layer.final_layer_norm(x_sa)
            fc1 = enc_layer.fc1(ln2)
            act = torch.nn.functional.gelu(fc1, approximate="none")
            ffn_out = enc_layer.fc2(act)

            x = x_sa + ffn_out

        save(f"{OUT_DIR}/encoder_block{i}_out.npy", x[0].cpu().numpy())

    final_manual = x[0].cpu().numpy()
    save(f"{OUT_DIR}/encoder_output_manual.npy", final_manual)

    # ----------------------------------------------------
    # HF encoder output
    # ----------------------------------------------------
    with torch.no_grad():
        enc_hf = model.model.encoder(mel).last_hidden_state

    final_hf = enc_hf[0].cpu().numpy()
    save(f"{OUT_DIR}/encoder_output_hf.npy", final_hf)

    diff = (enc_hf - x).abs().max().item()
    print(f"\nMax abs diff between manual and HF encoder: {diff:.6e}")

    # ====================================================
    # DECODER: actually transcribe + print tokens
    # ====================================================
    print("\n===== DECODER CHECK (HF) =====")

    # 1) Get the forced decoder prompt IDs that Whisper uses internally
    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    # forced_ids is a list of [position, token_id]; build the actual prompt sequence:
    prompt_token_ids = [tok_id for _, tok_id in forced_ids]
    prompt_len = len(prompt_token_ids)

    print("Forced prompt token IDs:", prompt_token_ids)
    print("Forced prompt decoded text:",
          tokenizer.decode(prompt_token_ids, skip_special_tokens=False))

    MAX_NEW = 64  # mirror your C++ autoregressive test

    # 2) Generate with proper Whisper-style config: use forced_decoder_ids
    with torch.no_grad():
        gen_ids = model.generate(
            input_features=mel,             # NOTE: use input_features for Whisper
            forced_decoder_ids=forced_ids,  # NOT decoder_input_ids
            max_new_tokens=MAX_NEW,
            do_sample=False,                # greedy
            num_beams=1,
        )  # shape: (1, total_len)

    gen_ids_list = gen_ids[0].tolist()
    print("\nAll generated token IDs:", gen_ids_list)

    # 3) Split prompt vs new tokens according to prompt_len
    gen_prompt = gen_ids_list[:prompt_len]
    gen_new = gen_ids_list[prompt_len:]

    print("\nPrompt part (from generate):", gen_prompt)
    print("Newly generated token IDs:  ", gen_new)

    # 4) Decode:
    # Full decoded string (good for sanity check)
    full_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    print("\nDecoded full sequence (no special token stripping):")
    print(full_text)

    # "True" transcription: decode only new tokens, skipping special tokens
    transcription = tokenizer.decode(gen_new, skip_special_tokens=True)
    print("\n===== FINAL TRANSCRIPTION (new tokens only) =====")
    print(transcription if transcription.strip() else "<EMPTY STRING>")
    print("=================================================")

    print("\n===== END DECODER CHECK =====")


if __name__ == "__main__":
    main()
