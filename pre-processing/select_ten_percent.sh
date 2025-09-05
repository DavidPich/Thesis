#!/bin/bash

# Define class
class_name="corrected"
# Define the source directory
select_directory="./pl_ten_percent/$class_name/"

# Define the destination directory where files will be moved
source_directory="spec_total/$class_name/"
destination_directory="spec_ten_percent/$class_name/"

# Define the number of files to move
num_files=62798

# Initialize a counter to keep track of the number of files moved
count=0

# Iterate over the files in the source directory
for file in "$select_directory"*
do
    # Check if the regular file exists
    if [ -f "$file" ]; then
        # Move the file
        echo "Copying $count / 62798"
        echo "${file##*/}"
        cp "$source_directory${file##*/}" "$destination_directory"
        
        # Increment the counter
        ((count++))
        
        # Check if the desired number of files have been moved
        if [ "$count" -eq "$num_files" ]; then
            echo "Copied $count files."
            break
        fi
    else
        echo "ERROR: File $source_directory${file##*/} not found!"
        break
    fi
done

echo "Copying of files is complete."
