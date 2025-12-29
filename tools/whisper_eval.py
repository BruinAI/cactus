import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import soundfile as sf
import re
import atexit
import unicodedata
import csv
from multiprocessing import Pool, cpu_count
sys.path.insert(0, "src")
from evaluate import load

from text_to_num import alpha2digit
from breame.spelling import get_american_spelling
import contractions

from cactus_ffi import (
    cactus_init,
    cactus_transcribe,
    cactus_reset,
    cactus_destroy
)

wer_metric = load("wer")

def normalize_text_en(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(hmm|mm|mhm|mmm|uh|um)\b', '', text)
    text = re.sub(r"\s+'", "'", text)
    text = contractions.fix(text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'\.(?!\d)', ' ', text)
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'[^\w\s\.\%\$\£\€\¥]+', ' ', text)
    text = alpha2digit(text, "en")
    text = re.sub(r'(\d+)\s+dollars?', r'$\1', text)
    text = re.sub(r'(\d+)\s+pounds?', r'£\1', text)
    text = re.sub(r'(\d+)\s+euros?', r'€\1', text)
    text = re.sub(r'(\d+)\s+yen', r'¥\1', text)
    words = text.split()
    words = [get_american_spelling(word) for word in words]
    text = ' '.join(words)
    text = re.sub(r'(?<!\d)[\$£€¥](?!\d)', ' ', text)
    text = re.sub(r'[^\w\s\$£€¥]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

worker_whisper = None
whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

def worker_init():
    global worker_whisper
    worker_whisper = cactus_init("../weights/whisper-small", context_size=448)
    atexit.register(worker_cleanup)

def worker_cleanup():
    global worker_whisper
    if worker_whisper is not None:
        cactus_destroy(worker_whisper)

def process_file(args):
    audio_file, reference = args

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        audio_data, sample_rate = sf.read(str(audio_file))
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

def load_librispeech(dataset_path, split):
    librispeech_path = Path(dataset_path) / split
    files_to_process = []

    for speaker_dir in sorted(librispeech_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if not trans_file.exists():
                continue

            transcriptions = {}
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcriptions[file_id] = text

            for flac_file in sorted(chapter_dir.glob("*.flac")):
                file_id = flac_file.stem
                if file_id in transcriptions:
                    files_to_process.append((flac_file, transcriptions[file_id]))

    return files_to_process

def load_voxpopuli(dataset_path, split):
    base_path = Path(dataset_path)
    tsv_file = base_path / "transcribed_data" / "en" / "asr_en.tsv"

    files_to_process = []

    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            if row['split'] != split:
                continue

            normed_text = row['normed_text'].strip()
            if not normed_text:
                continue

            session_id = row['session_id']
            audio_id = row['id_']
            year = session_id[:4]

            audio_file = base_path / "transcribed_data" / "en" / year / f"{session_id}-{audio_id}.ogg"

            if audio_file.exists():
                files_to_process.append((audio_file, normed_text))

    return files_to_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Whisper on ASR datasets")
    parser.add_argument("--dataset", type=str, default="librispeech",
                       choices=["librispeech", "voxpopuli"],
                       help="Dataset to evaluate on (default: librispeech)")
    parser.add_argument("--split", type=str, default="test-clean",
                       help="Dataset split to evaluate (default: test-clean)")
    parser.add_argument("--dataset-path", type=str,
                       default="/Users/satyajit/Downloads/LibriSpeech",
                       help="Path to dataset root")

    args = parser.parse_args()

    if args.dataset == "librispeech":
        files_to_process = load_librispeech(args.dataset_path, args.split)
        dataset_name = f"LibriSpeech {args.split}"
    elif args.dataset == "voxpopuli":
        files_to_process = load_voxpopuli(args.dataset_path, args.split)
        dataset_name = f"VoxPopuli {args.split}"

    print(f"Evaluating on {dataset_name}")
    print(f"Dataset path: {args.dataset_path}\n")

    references = []
    hypotheses = []

    print(f"Found {len(files_to_process)} files to process\n")

    num_workers = min(1, cpu_count())
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

    hypotheses = [normalize_text_en(h) for h in hypotheses]
    references = [normalize_text_en(r) for r in references]

    print(f"\nTotal files processed: {len(references)}")
    if errors > 0:
        print(f"Errors encountered: {errors}")

    wer_score = wer_metric.compute(predictions=hypotheses, references=references)
    print(f"Word Error Rate (WER): {wer_score:.4f} ({wer_score*100:.2f}%)")

    print("\nExample transcriptions:")
    for i in range(min(5, len(references))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference:  {references[i]}")
        print(f"Hypothesis: {hypotheses[i]}")
