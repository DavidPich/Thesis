import os
import soundfile as sf

def split_wav(wav_file,input_folder_path, output_folder_path):
    print(f"Progress: {progress_counter}/{num_wav_files} started!")
    progress_counter = progress_counter + 1
    # Load the .wav file
    file_path = os.path.join(input_folder_path, wav_file)
    data, samplerate = sf.read(file_path)

    # Calculate the number of segments
    segment_duration = 10  # seconds
    num_segments = int(len(data) / (samplerate * segment_duration))

    # Split the file into segments and save them
    for i in range(num_segments):
        start_sample = i * samplerate * segment_duration
        end_sample = (i + 1) * samplerate * segment_duration
        segment_data = data[start_sample:end_sample]

        # Create a new file name for the segment
        segment_file_name = f"{os.path.splitext(wav_file)[0]}_{i+1}.wav"
        segment_file_path = os.path.join(output_folder_path, segment_file_name)

        # Save the segment as a new .wav file
        sf.write(segment_file_path, segment_data, samplerate)

    print(f"File {wav_file} has been split into {num_segments} segments.")

def split_wav_files(input_folder_path, output_folder_path):
    # Get a list of all .wav files in the folder
    wav_files = [file for file in os.listdir(input_folder_path) if file.endswith('.wav')]
    num_wav_files = len(wav_files)
    print(num_wav_files)

    progress_counter = 1
    # Loop through each .wav file
    for wav_file in wav_files:
        print(f"Progress: {progress_counter}/{num_wav_files} started!")
        progress_counter = progress_counter + 1
        # Load the .wav file
        file_path = os.path.join(input_folder_path, wav_file)
        data, samplerate = sf.read(file_path)

        # Calculate the number of segments
        segment_duration = 10  # seconds
        num_segments = int(len(data) / (samplerate * segment_duration))

        # Split the file into segments and save them
        for i in range(num_segments):
            start_sample = i * samplerate * segment_duration
            end_sample = (i + 1) * samplerate * segment_duration
            segment_data = data[start_sample:end_sample]

            # Create a new file name for the segment
            segment_file_name = f"{os.path.splitext(wav_file)[0]}_{i+1}.wav"
            segment_file_path = os.path.join(output_folder_path, segment_file_name)

            # Save the segment as a new .wav file
            sf.write(segment_file_path, segment_data, samplerate)

        print(f"File {wav_file} has been split into {num_segments} segments.")



# Example usage
input_folder_path = 'data/wav'
output_folder_path = 'data/segmented'
split_wav_files(input_folder_path, output_folder_path)