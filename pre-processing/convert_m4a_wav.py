import os
from pydub import AudioSegment
from tqdm import tqdm

SOURCE_DIR = "data/m4a/"
DESTINATION_DIR = "data/wav/"

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def convert_m4a_to_wav(source_directory, destination_directory):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
     # Count all files in the source directory
    total_files = count_files(source_directory)
    print(f"Total files in source directory: {total_files}")

    # Iterate over all files in the source directory
    for filename in tqdm(os.listdir(source_directory), total=total_files, desc="Converting files"):
        if filename.endswith(".m4a"):
            # Define full file path
            m4a_file_path = os.path.join(source_directory, filename)
            
            # Load the m4a file
            audio = AudioSegment.from_file(m4a_file_path)
                        
            # Define the new file name for the .wav file
            wav_filename = f"{os.path.splitext(filename)[0]}.wav"
            wav_file_path = os.path.join(destination_directory, wav_filename)
            
            # Export as .wav
            audio.export(wav_file_path, format='wav')
            #print(f"Converted {filename} to {wav_filename}")

def main():
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"No directory: {SOURCE_DIR}")

    convert_m4a_to_wav(SOURCE_DIR, DESTINATION_DIR)



if __name__ == "__main__":
    main()
    
