import time
import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

AUDIO_FILE = "test.wav"
MODEL_NAME = "openai/whisper-medium"
WHISPER_SR = 16000

# All rows we will dump into profile.txt
profile_rows = []


def tensor_to_str(t: torch.Tensor, max_elems: int = 5) -> str:
    if t is None:
        return ""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.numel() == 0:
            return "[]"
        flat = t.reshape(-1)
        vals = ", ".join(f"{v:.4f}" for v in flat[:max_elems].tolist())
        if flat.numel() > max_elems:
            vals += ",..."
        return f"[{vals}]"
    return ""


def record_op(name: str,
              elapsed_ms: float,
              output: torch.Tensor | None,
              backend: str = "CPU",
              weights: torch.Tensor | None = None):
    if output is not None and isinstance(output, torch.Tensor):
        shape = list(output.shape)
        values_str = tensor_to_str(output)
    else:
        shape = []
        values_str = ""

    weights_str = tensor_to_str(weights) if weights is not None else ""

    profile_rows.append({
        "op": name,
        "time_ms": elapsed_ms,
        "shape": shape,
        "backend": backend,
        "values": values_str,
        "weights": weights_str,
    })


def profile_op(name: str, fn, backend: str = "CPU", weights: torch.Tensor | None = None):
    start = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    record_op(name, elapsed_ms, out, backend=backend, weights=weights)
    return out


def write_profile(path: str = "profile.txt"):
    with open(path, "w") as f:
        f.write("=== Graph Execution Profile ===\n")
        f.write("Operation      Time (ms)   Output Shape        Backend\n")
        f.write("------------------------------------------------------------\n")
        for row in profile_rows:
            op = row["op"]
            t = row["time_ms"]
            shape = "[" + ",".join(str(d) for d in row["shape"]) + "]" if row["shape"] else "[]"
            backend = row["backend"]
            line = f"{op:<14} {t:>9.3f}   {shape:<18} {backend}"
            if row["values"]:
                line += f"   values={row['values']}"
            if row["weights"]:
                line += f" weights={row['weights']}"
            f.write(line + "\n")
    print(f"✅ Wrote profile to {path}")


def main():
    print("Loading Whisper FP32 model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
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
    ).input_features.float()

    # (We don't save mel.npy anymore; if you want it, you can still add it back.)

    # ----------------------------------------------------
    # Encoder conv stem (conv1 / conv2) + transpose
    # ----------------------------------------------------
    with torch.no_grad():
        conv1 = profile_op("CONV1D_K3", lambda: model.model.encoder.conv1(mel))
        conv2 = profile_op("CONV1D_K3", lambda: model.model.encoder.conv2(conv1))

        # (1, d_model, T_enc) → (1, T_enc, d_model)
        conv2_T = profile_op("TRANSPOSE", lambda: conv2.transpose(1, 2))

    x = conv2_T  # (1, T_enc, d_model)
    T_enc = x.shape[1]
    D = x.shape[2]

    # ----------------------------------------------------
    # Positional embeddings + pre-LN (manual LN to mirror your C++ path)
    # ----------------------------------------------------
    with torch.no_grad():
        # Whisper encoder uses learned positional embeddings indexed by time
        pos_emb = profile_op(
            "ENC_POS_EMB",
            lambda: model.model.encoder.embed_positions.weight[:T_enc][None, :, :],
        )

        x = profile_op("ADD_POS", lambda: x + pos_emb)

        # Manual LayerNorm on CPU with NumPy to keep parity with your previous script
        w = model.model.encoder.layer_norm.weight.detach().cpu().numpy()
        b = model.model.encoder.layer_norm.bias.detach().cpu().numpy()

        def enc_layernorm_np_to_torch():
            enc_plus_pos = x[0].cpu().numpy()  # (T_enc, D)
            mean = enc_plus_pos.mean(-1, keepdims=True)
            var = enc_plus_pos.var(-1, keepdims=True)
            ln_np = (enc_plus_pos - mean) / np.sqrt(var + 1e-5)
            ln_np = ln_np * w + b
            return torch.tensor(ln_np, dtype=torch.float32).unsqueeze(0)

        x = profile_op("ENC_LAYERNORM", enc_layernorm_np_to_torch)

    # ----------------------------------------------------
    # Full encoder stack (manual, layer by layer)
    # ----------------------------------------------------
    with torch.no_grad():
        for i, enc_layer in enumerate(model.model.encoder.layers):
            def enc_block():
                ln1 = enc_layer.self_attn_layer_norm(x)
                sa_out = enc_layer.self_attn(ln1)[0]
                x_sa = x + sa_out

                ln2 = enc_layer.final_layer_norm(x_sa)
                fc1 = enc_layer.fc1(ln2)
                act = torch.nn.functional.gelu(fc1, approximate="none")
                ffn_out = enc_layer.fc2(act)

                return x_sa + ffn_out

            x = profile_op(f"ENC_BLOCK{i}", enc_block)

        final_manual = x  # (1, T_enc, D)
        profile_op("ENC_OUT_MANUAL", lambda: final_manual)

    # ----------------------------------------------------
    # HF encoder output (reference)
    # ----------------------------------------------------
    with torch.no_grad():
        enc_hf = model.model.encoder(mel).last_hidden_state  # (1, T_enc, D)

    final_hf = enc_hf
    profile_op("ENC_OUT_HF", lambda: final_hf)

    diff = (enc_hf - final_manual).abs().max().item()
    print(f"\nMax abs diff between manual and HF encoder: {diff:.6e}")

    # ====================================================
    # DECODER PREFILL PROFILE (forced prompt only)
    # ====================================================
    print("\n===== DECODER PREFILL PROFILE =====")

    # 1) Build forced decoder prompt IDs like Whisper generate()
    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    prompt_token_ids = [tok_id for _, tok_id in forced_ids]
    prompt_len = len(prompt_token_ids)

    print("Forced prompt token IDs:", prompt_token_ids)
    print("Forced prompt decoded text:",
          tokenizer.decode(prompt_token_ids, skip_special_tokens=False))

    with torch.no_grad():
        # Encoder hidden states for cross-attention
        encoder_hidden_states = enc_hf  # (1, T_enc, D)

        # Decoder input ids: forced prompt only (prefill)
        decoder_input_ids = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        decoder = model.model.decoder

        # ----- Embeddings -----
        def dec_embed_tokens():
            return decoder.embed_tokens(decoder_input_ids) * decoder.embed_scale

        dec_tok = profile_op("EMBED_DEC", dec_embed_tokens)

        def dec_pos_emb():
            # WhisperPositionalEmbedding ignores token values; only uses shape
            return decoder.embed_positions(decoder_input_ids)

        dec_pos = profile_op("POS_DEC", dec_pos_emb)

        x_dec = profile_op("ADD_POS_DEC", lambda: dec_tok + dec_pos)

        # ----- Decoder layers (no cache, prefill only) -----
        for layer_idx, dec_layer in enumerate(decoder.layers):
            # Self-attention block
            ln1 = profile_op(f"LN1_DEC{layer_idx}",
                             lambda dl=dec_layer, x=x_dec: dl.self_attn_layer_norm(x))

            sa_out = profile_op(
                f"SELF_ATTN_DEC{layer_idx}",
                # No past_key_value, no explicit mask → rely on is_causal if enabled
                lambda dl=dec_layer, ln=ln1: dl.self_attn(ln)[0],
            )

            x_sa = profile_op(
                f"ADD_DEC_SA{layer_idx}",
                lambda x=x_dec, sa=sa_out: x + sa,
            )

            # Cross-attention block (decoder → encoder)
            ln_ca = profile_op(
                f"LN_CA_DEC{layer_idx}",
                lambda dl=dec_layer, xs=x_sa: dl.encoder_attn_layer_norm(xs),
            )

            ca_out = profile_op(
                f"CROSS_ATTN_DEC{layer_idx}",
                lambda dl=dec_layer, ln=ln_ca, enc_h=encoder_hidden_states: dl.encoder_attn(
                    ln,
                    key_value_states=enc_h,
                )[0],
            )

            x_ca = profile_op(
                f"ADD_DEC_CA{layer_idx}",
                lambda xs=x_sa, co=ca_out: xs + co,
            )

            # FFN block
            ln2 = profile_op(
                f"LN2_DEC{layer_idx}",
                lambda dl=dec_layer, x=x_ca: dl.final_layer_norm(x),
            )

            fc1 = profile_op(
                f"FC1_DEC{layer_idx}",
                lambda dl=dec_layer, ln2=ln2: dl.fc1(ln2),
            )

            act = profile_op(
                f"GELU_DEC{layer_idx}",
                lambda fc1=fc1: torch.nn.functional.gelu(fc1, approximate="none"),
            )

            fc2 = profile_op(
                f"FC2_DEC{layer_idx}",
                lambda dl=dec_layer, act=act: dl.fc2(act),
            )

            x_dec = profile_op(
                f"ADD_DEC_FFN{layer_idx}",
                lambda x_ca=x_ca, fc2=fc2: x_ca + fc2,
            )

        # Final decoder layer norm
        dec_final = profile_op("DECODER_FINAL_LN", lambda: decoder.layer_norm(x_dec))

        # Project to logits (LM head) – this is what prefill would produce
        logits = profile_op("LOGITS", lambda: model.proj_out(dec_final))

    # ====================================================
    # HF DECODER CHECK (full generate, like your original script)
    # ====================================================
    print("\n===== DECODER CHECK (HF generate) =====")
    MAX_NEW = 64

    with torch.no_grad():
        gen_ids = model.generate(
            input_features=mel,
            forced_decoder_ids=forced_ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            num_beams=1,
        )

    gen_ids_list = gen_ids[0].tolist()
    print("\nAll generated token IDs:", gen_ids_list)

    gen_prompt = gen_ids_list[:prompt_len]
    gen_new = gen_ids_list[prompt_len:]

    print("\nPrompt part (from generate):", gen_prompt)
    print("Newly generated token IDs:  ", gen_new)

    full_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    print("\nDecoded full sequence (no special token stripping):")
    print(full_text)

    transcription = tokenizer.decode(gen_new, skip_special_tokens=True)
    print("\n===== FINAL TRANSCRIPTION (new tokens only) =====")
    print(transcription if transcription.strip() else "<EMPTY STRING>")
    print("=================================================")

    print("\n===== END DECODER CHECK =====")

    # Finally, dump everything to profile.txt
    write_profile("profile.txt")


if __name__ == "__main__":
    main()
