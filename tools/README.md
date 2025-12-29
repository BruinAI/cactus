# Whisper Evaluation on ASR Datasets

The whosper_eval.py script evaluates Whisper-small on the librispeech and voxpopuli datasets.

## Dataset Setup

### LibriSpeech

Download from: https://www.openslr.org/12

Expected structure:
```
LibriSpeech/
├── test-clean/
│   ├── 1089/
│   ├── 1188/
│   └── ...
└── test-other/
    ├── 116/
    ├── 1221/
    └── ...
```

### VoxPopuli

Download from: https://github.com/facebookresearch/voxpopuli
Follow the instructions to get the trasncribed text.

Expected structure:
```
VoxPopuli/
├── transcribed_data/
│   └── en/
│       ├── asr_en.tsv
│       ├── 2009/
│       ├── 2010/
│       └── ...
```

## Usage

### LibriSpeech

```bash
# Test-clean split
python whisper_eval.py \
  --dataset librispeech \
  --split test-clean \
  --dataset-path /path/to/LibriSpeech

# Test-other split
python whisper_eval.py \
  --dataset librispeech \
  --split test-other \
  --dataset-path /path/to/LibriSpeech
```

### VoxPopuli

```bash
# Test split
python whisper_eval.py \
  --dataset voxpopuli \
  --split test \
  --dataset-path /path/to/VoxPopuli

# Dev split
python whisper_eval.py \
  --dataset voxpopuli \
  --split dev \
  --dataset-path /path/to/VoxPopuli
```

## Results

Word Error Rate (WER) comparison:

| Dataset | Ours | OpenAI's |
|---------|------|----------|
| LibriSpeech test-clean | 3.89 | 3.4 |
| LibriSpeech test-other | 8.43 | 7.6 |
| VoxPopuli EN test | 8.55 | 8.3* |

\* OpenAI reports 8.3 on the full evaluation set; our result is on the test split only.

## Text Normalization

The script implements Whisper's 12-step english normalization pipeline:

1. Lowercase conversion
2. Remove brackets `[...]`
3. Remove parentheses `(...)`
4. Remove filler words (hmm, mm, uh, um)
5. Fix whitespace before apostrophes
6. Expand contractions using the `contractions` library
7. Remove commas between digits
8. Remove periods not followed by numbers
9. Remove diacritics
10. Normalize numbers and currency using `text2num`
11. Convert British to American spellings using `breame`
12. Remove symbols except in numeric expressions
13. Normalize whitespace

## Multiprocessing

Edit the script to adjust worker count (default: 1 for NPU):

```python
num_workers = min(1, cpu_count())
```