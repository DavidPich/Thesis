#!/usr/bin/python3
from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
import psola
import pandas as pd
import os
import concurrent.futures
import sys
from tqdm import tqdm 

SEMITONES_IN_OCTAVE = 12
INPUT_FOLDER_PATH_WAV = 'data/segmented'


def closest_pitch_smooth(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    
    midi_note = librosa.hz_to_midi(f0)

    rounded = np.subtract(np.round(midi_note), midi_note)
    rounded = np.divide(rounded, 2)
    rounded_midi = np.add(rounded, midi_note)
        
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    rounded_midi[nan_indices] = np.nan

    # Convert back to Hz.
    return librosa.midi_to_hz(rounded_midi)

def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""

    midi_note = np.round(librosa.hz_to_midi(f0))

    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan

    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)

def spectrogram(audio, sr, filename, correction_method):
    """
    Compute and save the spectrogram of an audio signal.

    Parameters:
    - audio (ndarray): The audio signal.
    - sr (int): The sampling rate of the audio signal.
    - filename (str): The path to the output file.
    - correction_method (str): The method used for pitch correction.
    """
    
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    _ , ax = plt.subplots()
    librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    ax.set_ylim([0, 2048])

    plt.ylabel('')
    plt.xlabel('')
    ax.set_axis_off() 
    plt.savefig(str(filename.parent / 'graph/spec' / correction_method / (filename.stem + '.png')), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def pitchLineGraph(f0, filename, correction_method):
    matplotlib.use('Agg')
    plt.figure()  # Create a new figure
    _, ax = plt.subplots()
    ax.set_ylim([0, 600])
    ax.set_xlim([0, 431])
    plt.plot(f0, label='', color='orange', linewidth=1)
    plt.ylabel('')
    plt.xlabel('')
    ax.set_axis_off()
    plt.savefig(str(filename.parent / 'graph/pl' / correction_method / (filename.stem + '.png')), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def autotune(audio, sr, filename=None):
   
    # Set some basis parameters for the pitch tracking (default values from the librosa library) 
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, _, _ = librosa.pyin(audio,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            sr=sr,
                            fmin=fmin,
                            fmax=fmax)

    # Exclude segments with less than 50% of pitch information
    f0[f0 > 600] = np.nan
    information_pct = ((1 - (np.count_nonzero(np.isnan(f0))/ f0.size)))
    
    if information_pct < 0.5:
        return


    # Apply the chosen adjustment strategy to the pitch    

    # Correct the pitch using the closest pitch method
    corrected_f0 = closest_pitch(f0)

    # Pitch-shifting using the PSOLA algorithm
    corrected_audio = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

    # Pitch tracking using the PYIN algorithm.
    corrected_audio_f0, _, _ = librosa.pyin(corrected_audio,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            sr=sr,
                                            fmin=fmin,
                                            fmax=fmax)
    
    # Save the corrected audio
    #filepath = filename.parent / (filename.stem + '_pc' + filename.suffix)
    #sf.write(str(filepath), corrected_audio, sr)

    # Correct the pitch using the smoothed closest pitch method
    corrected_f0_smooth = closest_pitch_smooth(f0)    

    # Pitch-shifting using the PSOLA algorithm
    corrected_smooth_audio = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0_smooth, fmin=fmin, fmax=fmax)
    
    # Pitch tracking using the PYIN algorithm.
    corrected_smooth_audio_f0, _, _ = librosa.pyin(corrected_smooth_audio,
                                                    frame_length=frame_length,
                                                    hop_length=hop_length,
                                                    sr=sr,
                                                    fmin=fmin,
                                                    fmax=fmax)
    
    # Save the smooth corrected audio
    filepath = filename.parent / (filename.stem + '_pcs' + filename.suffix)
    sf.write(str(filepath), corrected_smooth_audio, sr)
    
    # Plot the pitch line
    pitchLineGraph(f0, filename, 'original')
    pitchLineGraph(corrected_audio_f0, filename, 'corrected')
    pitchLineGraph(corrected_smooth_audio_f0, filename, 'smoothed')

    ## Plot the spectrogram
    spectrogram(audio,sr,filename,'original')
    spectrogram(corrected_audio,sr,filename,'corrected')
    spectrogram(corrected_smooth_audio,sr,filename,'smoothed')

def create_directory_structure():
    if not os.path.exists(INPUT_FOLDER_PATH_WAV):
        os.makedirs(INPUT_FOLDER_PATH_WAV)
        print(f"Created directory: {INPUT_FOLDER_PATH_WAV}")

    if not os.path.exists("data/segmented/graph/pl/original/"): os.makedirs("data/segmented/graph/pl/original/")
    if not os.path.exists("data/segmented/graph/pl/corrected/"): os.makedirs("data/segmented/graph/pl/corrected/") 
    if not os.path.exists("data/segmented/graph/pl/smoothed/"): os.makedirs("data/segmented/graph/pl/smoothed/")
    if not os.path.exists("data/segmented/graph/spec/original/"): os.makedirs("data/segmented/graph/spec/original/")
    if not os.path.exists("data/segmented/graph/spec/corrected/"): os.makedirs("data/segmented/graph/spec/corrected/") 
    if not os.path.exists("data/segmented/graph/spec/smoothed/"): os.makedirs("data/segmented/graph/spec/smoothed/") 

def process_file(wav_file):
    file_path = Path(os.path.join(INPUT_FOLDER_PATH_WAV, wav_file))
    # Load the audio file.
    y, sr = librosa.load(str(file_path), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Without args
    autotune(y, sr, file_path)

def process_files_in_parallel(file_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) for filename in file_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # Handle result if needed
            except Exception as e:
                print(f"Exception occurred: {e}")

def main():
    create_directory_structure()
    wav_files = [file for file in os.listdir(INPUT_FOLDER_PATH_WAV) if file.endswith('.wav')]
    num_wav_files = len(wav_files)

    
    print(f"{num_wav_files} files to process")
    #process_files_in_parallel(wav_files)

    for wav_file in tqdm(wav_files):
        process_file(wav_file)

    
if __name__=='__main__':
    main()
    