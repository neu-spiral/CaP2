#!/bin/bash

# Check if directory and substring are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <directory> <substring_to_remove>"
    exit 1
fi

# Assign command-line arguments to variables
directory=$1
substring=$2

# Loop through all files in the specified directory
for file in "$directory"/*; do
    # Check if it's a file (ignore directories)
    if [ -f "$file" ]; then
        # Get the file's basename (remove directory path)
        filename=$(basename "$file")

        # Remove the substring from the filename
        new_filename="${filename//$substring/}"

        # Rename the file if the new filename is different
        if [ "$filename" != "$new_filename" ]; then
            mv "$file" "$directory/$new_filename"
            echo "Renamed: $filename -> $new_filename"
        fi
    fi
done

echo "File renaming complete."
