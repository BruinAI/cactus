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
    # Decoder prompt tokens
    # ----------------------------------------------------
    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    decoder_input_ids = torch.tensor([[i[1] for i in forced_ids]], dtype=torch.int64)
    save(f"{OUT_DIR}/decoder_input_tokens.npy", decoder_input_ids.cpu().numpy())

    # ----------------------------------------------------
    # Conv1 + Conv2
    # ----------------------------------------------------
    with torch.no_grad():
        conv1 = model.model.encoder.conv1(mel)
        conv2 = model.model.encoder.conv2(conv1)

    save(f"{OUT_DIR}/conv1_out.npy", conv1.cpu().numpy())
    save(f"{OUT_DIR}/conv2_out.npy", conv2.cpu().numpy())

    # Transpose (Whisper does this inside HF)
    conv2_T = conv2.transpose(1, 2)    # (1, T, 1024)
    conv2_T_np = conv2_T[0].cpu().numpy()
    save(f"{OUT_DIR}/conv2_T.npy", conv2_T_np)

    T_enc = conv2_T.shape[1]
    D = conv2_T.shape[2]

    # ----------------------------------------------------
    # Positional embeddings
    # ----------------------------------------------------
    pos_emb = model.model.encoder.embed_positions.weight[:T_enc].detach().cpu().numpy()
    save(f"{OUT_DIR}/encoder_pos_emb.npy", pos_emb)

    enc_plus_pos = conv2_T_np + pos_emb
    save(f"{OUT_DIR}/encoder_pos_added.npy", enc_plus_pos)

    # ----------------------------------------------------
    # Pre-transformer LayerNorm
    # ----------------------------------------------------
    w = model.model.encoder.layer_norm.weight.detach().cpu().numpy()
    b = model.model.encoder.layer_norm.bias.detach().cpu().numpy()

    mean = enc_plus_pos.mean(-1, keepdims=True)
    var = enc_plus_pos.var(-1, keepdims=True)

    ln = (enc_plus_pos - mean) / np.sqrt(var + 1e-5)
    ln = ln * w + b
    save(f"{OUT_DIR}/encoder_post_norm.npy", ln)

    # ----------------------------------------------------
    # ******* 🔥 NEW: Extract encoder block 0 ********
    # ----------------------------------------------------
    x = torch.tensor(ln, dtype=torch.float32).unsqueeze(0)   # (1, T, 1024)
    enc0 = model.model.encoder.layers[0]

    # LN before SA
    with torch.no_grad():
        block0_ln1 = enc0.self_attn_layer_norm(x)        # (1, T, 1024)

    save(f"{OUT_DIR}/encoder_block0_ln1.npy", block0_ln1[0].cpu().numpy())

    # Self-attention
    with torch.no_grad():
        sa_out = enc0.self_attn(block0_ln1, attention_mask=None)[0]   # (1, T, 1024)

    save(f"{OUT_DIR}/encoder_block0_sa_out.npy", sa_out[0].cpu().numpy())

    # Residual add
    x_sa = x + sa_out

    # FFN
    with torch.no_grad():
        block0_ln2 = enc0.final_layer_norm(x_sa)

    # FFN = fc2( GELU( fc1(x) ) )
    with torch.no_grad():
        ffn_fc1 = enc0.fc1(block0_ln2)
        ffn_act = torch.nn.functional.gelu(ffn_fc1)
        ffn_out = enc0.fc2(ffn_act)

    # Block output: residual add
    block0_out = x_sa + ffn_out
    save(f"{OUT_DIR}/encoder_block0_out.npy", block0_out[0].cpu().numpy())

    # ----------------------------------------------------
    # Full encoder output (after all 24 blocks)
    # ----------------------------------------------------
    with torch.no_grad():
        final_enc = model.model.encoder(input_features=mel).last_hidden_state

    save(f"{OUT_DIR}/encoder_output_final.npy", final_enc.cpu().numpy())

    # ----------------------------------------------------
    # Logits + transcription
    # ----------------------------------------------------
    with torch.no_grad():
        out = model(input_features=mel, decoder_input_ids=decoder_input_ids)
        save(f"{OUT_DIR}/logits_ref.npy", out.logits.cpu().numpy())

    with torch.no_grad():
        gen = model.generate(mel, task="transcribe", language="en", max_length=128)

    print("\nTRANSCRIPT:", tokenizer.decode(gen[0], skip_special_tokens=True))
    print("\nAll reference tensors saved!")


if __name__ == "__main__":
    main()
