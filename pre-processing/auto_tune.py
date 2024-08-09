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

SEMITONES_IN_OCTAVE = 12
INPUT_FOLDER_PATH = 'Flo/data/segmented'
OUTPUT_FOLDER_PATH = 'data/segmented_pc'


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
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    #time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
    log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    #fig.colorbar(img, ax=ax, format="%+2.f dB")
    #ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
    #ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
    #ax.legend(loc='upper right')
    ax.set_ylim([0, 2048])

    plt.ylabel('')
    plt.xlabel('')
    ax.set_axis_off() 
    #plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(filename.parent / 'spec' / correction_method / (filename.stem + '.png')), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def pitchLineGraph(f, filename, correction_method):
    matplotlib.use('Agg')
    plt.figure()  # Create a new figure
    _, ax = plt.subplots()
    ax.set_ylim([0, 600])
    ax.set_xlim([0, 431])
    plt.plot(f, label='original pitch', color='orange', linewidth=1)
    plt.ylabel('')
    plt.xlabel('')
    ax.set_axis_off()
    plt.savefig(str(filename.parent / 'pl' / correction_method / (filename.stem + '.png')), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def autotune(audio, sr, plot=False, filename=None):
    global lowInfo

    # Set some basis parameters.
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

    # Excluse empty graphs


    # Apply the chosen adjustment strategy to the pitch.
    f0[f0 > 600] = np.nan
    information_pct = ((1 - (np.count_nonzero(np.isnan(f0))/ f0.size)))
    print(f"Information percentage: {information_pct}")
    print(f0.size)
    if information_pct < 0.5:
        lowInfo += 1
        print(f"File {filename} has low information percentage: {information_pct}")
        #sys.exit(1)
        return
    
    corrected_f0 = closest_pitch(f0)
    corrected_f0_smooth = closest_pitch_smooth(f0)

    corrected_audio = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)
    corrected_smooth_audio = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0_smooth, fmin=fmin, fmax=fmax)

    corrected_audio_f0, _, _ = librosa.pyin(corrected_audio,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            sr=sr,
                            fmin=fmin,
                            fmax=fmax)
    
    corrected_smooth_audio_f0, _, _ = librosa.pyin(corrected_smooth_audio,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            sr=sr,
                            fmin=fmin,
                            fmax=fmax)
    
    #print(f"Original {f0}")
    #print(f"Shifted {corrected_f0}f")

    if plot:
        # To just plot the spectrogram without the pitchlines, if needed remove comment
        # Plot the pitch trajectory of the original pitch
        #plt.figure()  # Create a new figure 

        # Plot the pitch line
        #pitchLineGraph(f0, filename, 'original')
        #pitchLineGraph(corrected_audio_f0, filename, 'corrected')
        #pitchLineGraph(corrected_smooth_audio_f0, filename, 'smoothed')

        # Plot the spectrogram
        spectrogram(audio,sr,filename,'original')
        spectrogram(corrected_audio,sr,filename,'corrected')
        spectrogram(corrected_smooth_audio,sr,filename,'smoothed')


        filepath = filename.parent / (filename.stem + '_pc' + filename.suffix)
        #sf.write(str(filepath), corrected_audio, sr)


    return
    # Pitch-shifting using the PSOLA algorithm.
    # return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def create_directory_structure():
    if not os.path.exists("Flo/data/segmented/pl/original/"): os.makedirs("Flo/data/segmented/pl/original/")
    if not os.path.exists("Flo/data/segmented/pl/corrected/"): os.makedirs("Flo/data/segmented/pl/corrected/") 
    if not os.path.exists("Flo/data/segmented/pl/smoothed/"): os.makedirs("Flo/data/segmented/pl/smoothed/")
    if not os.path.exists("Flo/data/segmented/spec/original/"): os.makedirs("Flo/data/segmented/spec/original/")
    if not os.path.exists("Flo/data/segmented/spec/corrected/"): os.makedirs("Flo/data/segmented/spec/corrected/") 
    if not os.path.exists("Flo/data/segmented/spec/smoothed/"): os.makedirs("Flo/data/segmented/spec/smoothed/") 
    if not os.path.exists("Flo/data/segmented/pl/original/"): os.makedirs("Flo/data/segmented/pl/original/")
    if not os.path.exists("Flo/data/segmented/pl/corrected/"): os.makedirs("Flo/data/segmented/pl/corrected/") 
    if not os.path.exists("Flo/data/segmented/pl/smoothed/"): os.makedirs("Flo/data/segmented/pl/smoothed/")
    if not os.path.exists("Flo/data/segmented/spec/original/"): os.makedirs("Flo/data/segmented/spec/original/")
    if not os.path.exists("Flo/data/segmented/spec/corrected/"): os.makedirs("Flo/data/segmented/spec/corrected/") 
    if not os.path.exists("Flo/data/segmented/spec/smoothed/"): os.makedirs("Flo/data/segmented/spec/smoothed/") 

def process_file(wav_file):
    file_path = Path(os.path.join(INPUT_FOLDER_PATH, wav_file))
    # Load the audio file.
    y, sr = librosa.load(str(file_path), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Without args
    autotune(y, sr, True, file_path)
    #print(f"File: {file_path}, highest pitch: {highestPitch}, lowest pitch: {lowestPitch}")
    #print(f"Low Information: {lowInfo}")
    #print(f"Under 100: {under100} Between 100 and 300: {between100and300} Between 300 and 600: {between300and600} Between 600 and 1200: {between600and1200} Over 1200: {over12}")

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
    wav_files = [file for file in os.listdir(INPUT_FOLDER_PATH) if file.endswith('.wav')]
    num_wav_files = len(wav_files)
    print(num_wav_files)
    process_files_in_parallel(wav_files)
    
    
    #No multithreading
    #for wav_file in wav_files:
    #    process_file(wav_file)
    
    #single file
    #process_file('67982109_31286272_14.wav')


    
if __name__=='__main__':
    main()
    