import os
import soundfile as sf
import multiprocessing
import tqdm

INPUT_FOLDER_PATH = 'data/wav'
OUTPUT_FOLDER_PATH = 'data/segmented'
SEGMENT_DURATION = 10  # seconds

def split_wav(wav_file):
    # Load the .wav file
    file_path = os.path.join(INPUT_FOLDER_PATH, wav_file)
    data, samplerate = sf.read(file_path)

    # Calculate the number of segments
    num_segments = int(len(data) / (samplerate * SEGMENT_DURATION))

    # Split the file into segments and save them
    for i in range(num_segments):
        start_sample = i * samplerate * SEGMENT_DURATION
        end_sample = (i + 1) * samplerate * SEGMENT_DURATION
        segment_data = data[start_sample:end_sample]

        # Create a new file name for the segment
        segment_file_name = f"{os.path.splitext(wav_file)[0]}_{i+1}.wav"
        segment_file_path = os.path.join(OUTPUT_FOLDER_PATH, segment_file_name)

        # Save the segment as a new .wav file
        sf.write(segment_file_path, segment_data, samplerate)

    #print(f"File {wav_file} has been split into {num_segments} segments.")

def create_directory_structure():
    if not os.path.exists("data/segmented/"): os.makedirs("data/segmented/")

def main():
    # Get a list of all .wav files in the folder
    create_directory_structure()
    wav_files = [file for file in os.listdir(INPUT_FOLDER_PATH) if file.endswith('.wav')]
    num_wav_files = len(wav_files)
    print(num_wav_files)

    # Loop through each .wav file
    
    # creating a pool object 
    p = multiprocessing.Pool() 
    # map list to target function
    for _ in tqdm.tqdm(p.imap(split_wav,wav_files), total=num_wav_files):
        pass


if __name__ == '__main__':
    main()