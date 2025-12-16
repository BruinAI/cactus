import sys
import json
from pathlib import Path
from tqdm import tqdm
import tempfile
import soundfile as sf
import re
sys.path.insert(0, "src")
from evaluate import load
import librosa

from cactus_ffi import (
    cactus_init,
    cactus_complete,
    cactus_transcribe,
    cactus_embed,
    cactus_image_embed,
    cactus_audio_embed,
    cactus_reset,
    cactus_destroy
)

# Load WER metric
wer_metric = load("wer")

def normalize_text(text):
    """Normalize text for WER calculation by removing punctuation and extra whitespace."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

print("\nLoading whisper-small...")
whisper = cactus_init("../weights/whisper-small", context_size=448)
whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Path to LibriSpeech test-clean dataset
librispeech_path = Path("/Users/satyajit/Downloads/LibriSpeech/test-clean")

references = []
hypotheses = []

# Collect all FLAC files and their transcriptions
files_to_process = []
for speaker_dir in sorted(librispeech_path.iterdir()):
    if not speaker_dir.is_dir():
        continue

    for chapter_dir in sorted(speaker_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        # Find the transcription file
        trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
        if not trans_file.exists():
            continue

        # Read transcriptions
        transcriptions = {}
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcriptions[file_id] = text

        # Collect FLAC files with their references
        for flac_file in sorted(chapter_dir.glob("*.flac")):
            file_id = flac_file.stem
            if file_id in transcriptions:
                files_to_process.append((flac_file, transcriptions[file_id]))

print(f"Found {len(files_to_process)} files to process\n")

# Process with progress bar
with tempfile.TemporaryDirectory() as tmpdir:
    for flac_file, reference in tqdm(files_to_process, desc="Transcribing"):
        # Convert FLAC to WAV (16kHz mono for Whisper)
        wav_path = Path(tmpdir) / f"{flac_file.stem}.wav"
        audio_data, sample_rate = sf.read(str(flac_file))

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"sampling rate not right: {sample_rate}")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        sf.write(str(wav_path), audio_data, sample_rate)

        # Clear KV cache before transcribing
        cactus_reset(whisper)

        # Transcribe the audio
        response = cactus_transcribe(whisper, str(wav_path), prompt=whisper_prompt)
        result = json.loads(response)

        if not result.get("success", False):
            print(f"\nError transcribing {flac_file.name}: {result.get('error', 'Unknown error')}")
            continue

        transcription = result.get("response", "").strip()

        # Remove the <|startoftranscript|> token from the end if present
        transcription = transcription.removesuffix("<|startoftranscript|>").strip()

        # Collect results (normalized for WER calculation)
        references.append(normalize_text(reference))
        hypotheses.append(normalize_text(transcription))

print(f"\nTotal files processed: {len(references)}")

# Calculate WER
wer_score = wer_metric.compute(predictions=hypotheses, references=references)
print(f"Word Error Rate (WER): {wer_score:.4f} ({wer_score*100:.2f}%)")

# Show some examples
print("\nExample transcriptions:")
for i in range(min(5, len(references))):
    print(f"\n--- Sample {i+1} ---")
    print(f"Reference:  {references[i]}")
    print(f"Hypothesis: {hypotheses[i]}")

cactus_destroy(whisper)