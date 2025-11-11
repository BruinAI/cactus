import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------
# CONFIG
# ---------------------------
AUDIO_FILE = "test.wav"
MODEL_NAME = "openai/whisper-medium"
OUT_DIR = "./"

# Whisper exact STFT/mel parameters
WHISPER_SR = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

def save_npy(path, arr):
    print(f"✅ Saved {path}  shape={arr.shape}  dtype={arr.dtype}")
    np.save(path, arr)

def main():
    print("Loading Whisper-medium FP16...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cpu"
    )

    # ----------------------------------------------------
    # Load audio
    # ----------------------------------------------------
    print(f"Loading audio: {AUDIO_FILE}")
    audio, sr = librosa.load(AUDIO_FILE, sr=WHISPER_SR)

    # ----------------------------------------------------
    # Compute Whisper mel spectrogram exactly
    # ----------------------------------------------------
    print("Computing mel spectrogram...")
    mel = processor.feature_extractor(
        audio,
        sampling_rate=WHISPER_SR,
        return_tensors="pt"
    ).input_features  # shape: (1, 80, T)

    mel_np = mel.cpu().numpy().astype(np.float32)
    save_npy(f"{OUT_DIR}/mel.npy", mel_np)

    # ----------------------------------------------------
    # Prepare decoder BOS tokens
    # ----------------------------------------------------
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    # Convert to array of ints
    decoder_input_ids = torch.tensor([i[1] for i in forced_decoder_ids]).unsqueeze(0)

    dec_np = decoder_input_ids.cpu().numpy().astype(np.int32)
    save_npy(f"{OUT_DIR}/decoder_input_tokens.npy", dec_np)

    # ----------------------------------------------------
    # Run Whisper forward → final logits
    # ----------------------------------------------------
    print("Running Whisper forward pass...")
    with torch.no_grad():
        out = model(
            input_features=mel.half(),
            decoder_input_ids=decoder_input_ids
        )

    logits = out.logits  # shape: (1, seq, vocab)
    logits_np = logits.cpu().numpy().astype(np.float32)

    save_npy(f"{OUT_DIR}/logits_ref.npy", logits_np)

    # Optional: save encoder hidden states
    enc = model.model.encoder(input_features=mel.half())
    enc_np = enc.last_hidden_state.detach().cpu().numpy().astype(np.float32)
    save_npy(f"{OUT_DIR}/encoder_hidden.npy", enc_np)

    print("\n✅ All reference data generated successfully!")

if __name__ == "__main__":
    main()
