import sys
import json
from pathlib import Path
from tqdm import tqdm
import tempfile
import soundfile as sf
import re
from multiprocessing import Pool, cpu_count
sys.path.insert(0, "src")
from evaluate import load

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

# Global worker state
worker_whisper = None
whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

def worker_init():
    """Initialize whisper model in each worker process."""
    global worker_whisper
    worker_whisper = cactus_init("../weights/whisper-small", context_size=448)

def worker_cleanup():
    """Cleanup whisper model in each worker process."""
    global worker_whisper
    if worker_whisper is not None:
        cactus_destroy(worker_whisper)

def process_file(args):
    flac_file, reference = args

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        audio_data, sample_rate = sf.read(str(flac_file))
        sf.write(wav_path, audio_data, sample_rate)

        cactus_reset(worker_whisper)

        response = cactus_transcribe(worker_whisper, wav_path, prompt=whisper_prompt)
        result = json.loads(response)

        Path(wav_path).unlink()

        if not result.get("success", False):
            return None, None, result.get('error', 'Unknown error')

        transcription = result.get("response", "").strip().removesuffix("<|startoftranscript|>").strip()

        return transcription, reference, None

    except Exception as e:
        return None, None, str(e)

if __name__ == '__main__':
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

    num_workers = min(8, cpu_count())
    print(f"Using {num_workers} worker processes\n")

    with Pool(processes=num_workers, initializer=worker_init) as pool:
        results = list(tqdm(
            pool.imap(process_file, files_to_process),
            total=len(files_to_process),
            desc="Transcribing"
        ))

    errors = 0
    for transcription, reference, error in results:
        if error:
            errors += 1
            continue
        hypotheses.append(transcription)
        references.append(reference)

    hypotheses = [normalize_text(h) for h in hypotheses]
    references = [normalize_text(r) for r in references]

    print(f"\nTotal files processed: {len(references)}")
    if errors > 0:
        print(f"Errors encountered: {errors}")

    # Calculate WER
    wer_score = wer_metric.compute(predictions=hypotheses, references=references)
    print(f"Word Error Rate (WER): {wer_score:.4f} ({wer_score*100:.2f}%)")

    # Show some examples
    print("\nExample transcriptions:")
    for i in range(min(5, len(references))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference:  {references[i]}")
        print(f"Hypothesis: {hypotheses[i]}")