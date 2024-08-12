# Thesis

This repository can be used to create a dataset of pitch line graphs and spectrograms in three categories.

- original
- corrected
- smoothed

## Usage

1. **Convert the files**:
   - Convert the `.m4a` files to `.wav` files.

2. **Segmentation**:
   - Divide the `.wav` files into 10-second segments.

3. **Apply pitch correction**
   - Apply the pitch correction on the vocal files

3. **Create the graphs**:
   - Create the pitch line graphs and spectrograms based on the 10-second segments.

## Analysis

Additionally, there is a script to analyze the files and extract a distribution of the frequencies to justify decisions.

## Scripts

- `convert_m4a_to_wav.py`: Script to convert `.m4a` files to `.wav` files.
- `segment_wav_files.py`: Script to segment the `.wav` files into 10-second segments.
- `auto_tune.py`: Script to apply the pitch correction and create the pitch line graphs and spectrograms.
- `analyze_frequencies.py`: Script to analyze the frequency distribution.

## Installation

Make sure to install all dependencies using the `requirements.txt` file:

```sh
pip install -r requirements.txt