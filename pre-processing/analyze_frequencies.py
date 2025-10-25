import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm

INPUT_FOLDER_PATH = 'data/segmented'

highestPitch = 0
lowestPitch = 100000
under100 = 0
between100and300 = 0
between300and600 = 0
between600and1200 = 0
over12 = 0
lowInfo = 0

# AI was used to generate a starting template for this function which was then modified and expanded
def count_Frequencies(wav_file):
    global highestPitch
    global lowestPitch
    global over12
    global under100
    global between100and300
    global between300and600
    global between600and1200
    global lowInfo

    file_path = Path(os.path.join(INPUT_FOLDER_PATH, wav_file))
    y, sr = librosa.load(str(file_path), sr=None, mono=False)

    if y.ndim > 1:
        y = y[0, :]

    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    f0, _, _ = librosa.pyin(y,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            sr=sr,
                            fmin=fmin,
                            fmax=fmax)

    # Filter NaN values
    f0 = f0[~np.isnan(f0)]

    if f0.size == 0:
        return

    highestPitch = max(highestPitch, np.max(f0[f0 < 1200], initial=highestPitch))
    lowestPitch = min(lowestPitch, np.min(f0, initial=lowestPitch))

    over12 += np.sum(f0 > 1200)
    under100 += np.sum(f0 < 100)
    between100and300 += np.sum((f0 >= 100) & (f0 < 300))
    between300and600 += np.sum((f0 >= 300) & (f0 < 600))
    between600and1200 += np.sum((f0 >= 600) & (f0 < 1200))
        
def main():
    wav_files = [file for file in os.listdir(INPUT_FOLDER_PATH) if file.endswith('.wav')]
    num_wav_files = len(wav_files)
    print(num_wav_files)

    for wav_file in tqdm(wav_files):
        count_Frequencies(wav_file)

    print(f"highest pitch: {highestPitch}, lowest pitch: {lowestPitch}")    
    print(f"Under 100: {under100} Between 100 and 300: {between100and300} Between 300 and 600: {between300and600} Between 600 and 1200: {between600and1200} Over 1200: {over12}")


if __name__=='__main__':
    main()
    