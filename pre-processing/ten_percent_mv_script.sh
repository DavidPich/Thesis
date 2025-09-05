#!/bin/bash

# Define class
class_name="original"
# Define the source directory
source_directory="./$class_name/"

# Define the destination directory where files will be moved
destination_directory="./filtered_$class_name/"

# Define the number of files to move
num_files=62798

# Initialize a counter to keep track of the number of files moved
count=0

# Iterate over the files in the source directory
for file in "$source_directory"*
do
    # Check if the regular file exists
    if [ -f "$file" ]; then
        # Move the file
        echo "Moving $count / 62798"
        mv "$file" "$destination_directory"
        
        # Increment the counter
        ((count++))
        
        # Check if the desired number of files have been moved
        if [ "$count" -eq "$num_files" ]; then
            echo "Moved $count files."
            break
        fi
    fi
done

echo "Moving of files is complete."
