# Audio Pre-processing and CNN Classification (Original | Corrected | Smoothed)

This thesis converts audio recordings into an image dataset of pitch line graphs and spectrograms in three classes and then trains CNNs (VGG/ResNet/RegNet) to classify them.

- original
- corrected (pitch snapped to the nearest MIDI note)
- smooth corrected (softer correction)

Pipeline: 
m4a → wav → segmentation (10 s) → pitch detection and correction → image generation → CNN training.

## Prerequisites

- Python 3.9–3.11 recommended
- ffmpeg (required for m4a → wav)
  - macOS: `brew install ffmpeg`
- Optional GPU (CUDA) or Apple Silicon (MPS). Scripts auto-detect CUDA/MPS and fall back to CPU.

## Installation

```bash
pip install -r requirements.txt
```

## Scripts

Pre-processing:
- `convert_m4a_wav.py`: Script to convert `.m4a` files to `.wav` files.
- `segment_wav_files.py`: Script to segment the `.wav` files into 10-second segments.
- `auto_tune.py`: Script to apply the pitch correction and create the pitch line graphs and spectrograms.
    - pitch line images: `data/segmented/graph/pl/{original,corrected,smoothed}`
    - spectrograms: `data/segmented/graph/spec/{original,corrected,smoothed}`
- `analyze_frequencies.py`: Script to analyze the frequency distribution.

Classification:
- `classification_script_VGG.py` – train VGG16
- `classification_script_ResNet.py` – train ResNet50
- `classification_script_RegNet.py` – train RegNetY-3.2GF

Data folders:
- `data/m4a/` – input files
- `data/wav/` – converted WAVs
- `data/segmented/` – 10 seconds segments
- `data/segmented/graph/pl/` - pitch line data
- `data/segmented/graph/spec/` - spectrogram data

Outputs:
- `results/` – per-model training CSV logs and test metrics (TXT)
  - VGG: `results/VGG/`
  - RegNet: `results/RegNet/`
  - ResNet: `results/ResNet/`

## Usage

1) m4a → wav

- Put `.m4a` files into `data/m4a/` and run:

```bash
python pre-processing/convert_m4a_wav.py
```

Result: WAVs in `data/wav/`.

2) Segmentation (10 s)

```bash
python pre-processing/segment_wav_files.py
```

Result: WAV segments in `data/segmented/`.

3) Pitch correction and image generation

```bash
python pre-processing/auto_tune.py
```

Results:
- Pitch line plots: `data/segmented/graph/pl/{original,corrected,smoothed}/*.png`
- Spectrograms: `data/segmented/graph/spec/{original,corrected,smoothed}/*.png`

4) Train the models

- VGG16:

```bash
python classification_script_VGG.py
```

- ResNet50:

```bash
python classification_script_ResNet.py
```

- RegNetY-3.2GF:

```bash
python classification_script_RegNet.py
```

Outputs:
- Per-epoch CSV logs (loss, validation accuracy) under `results/.../train_YYYYMMDD_HHMMSS.csv`
- Test metrics (accuracy, F1, precision, recall, confusion matrix) in `results/.../test_data_*.txt`



## Analysis
Additionally, there is a script to analyze the files and extract a distribution of the frequencies for sanity checks and design decisions.

```bash
python pre-processing/analyze_frequencies.py
```

Console output prints min/max and counts across frequency bands.