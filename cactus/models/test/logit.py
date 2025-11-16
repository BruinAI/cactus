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

    audio, _ = librosa.load(AUDIO_FILE, sr=16000)
    mel = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.float()

    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    prompt = torch.tensor([[i[1] for i in forced]], dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_features=mel,
            decoder_input_ids=prompt,
            return_dict=True,
        )
        logits = out.logits[0, -1].cpu().numpy()

    np.save("hf_prefill_logits.npy", logits)
    print("Saved hf_prefill_logits.npy, shape:", logits.shape)

    # Print top-10 tokens
    top_ids = logits.argsort()[-10:][::-1]
    for tid in top_ids:
        print(tid, logits[tid])

if __name__ == "__main__":
    main()
