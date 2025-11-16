import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

AUDIO_FILE = "test.wav"
MODEL_NAME = "openai/whisper-medium"

def main():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    # Same audio -> mel
    audio, _ = librosa.load(AUDIO_FILE, sr=16000)
    mel = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.float()

    # Same forced prompt tokens
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    prompt = torch.tensor([[i[1] for i in forced]], dtype=torch.long)

    with torch.no_grad():
        # run encoder
        enc_out = model.model.encoder(
            mel,
            return_dict=True
        ).last_hidden_state  # (1, T_enc, 1024)

        # run decoder explicitly
        dec_out = model.model.decoder(
            input_ids=prompt,
            encoder_hidden_states=enc_out,
            use_cache=False,
            return_dict=True,
        )

    # HF decoder output is *already* post-final-layernorm
    dec_hidden = dec_out.last_hidden_state  # (1, 3, 1024)

    # Get entire 3x1024 (so it matches your dec_norm tensor)
    dec_np = dec_hidden[0].cpu().numpy()
    np.save("decoder_post_norm_ref.npy", dec_np)
    print("Saved decoder_post_norm_ref.npy", dec_np.shape)

    # Also print first 2 rows 5 cols for eyeballing
    for r in range(2):
        print("row", r, ":", dec_np[r, :5])

if __name__ == "__main__":
    main()
