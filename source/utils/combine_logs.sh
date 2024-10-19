#!/bin/bash

# Check if the parent folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_folder>"
    exit 1
fi

# Parent folder provided as first argument
parent_folder=$1

# Find all subfolders (within a1, a2, a3) assuming the structure is the same
# Use a1 as a reference to get the subfolders
first_subfolder=$(find "$parent_folder" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$first_subfolder" ]; then
    echo "No subfolders found in $parent_folder."
    exit 1
fi

# Loop through each subfolder in the first reference folder (e.g., a1)
for subfolder in "$first_subfolder"/*; do
    subfolder_name=$(basename "$subfolder")

    # Create a combined folder for each subfolder (e.g., combined_b)
    combined_folder="$parent_folder/$subfolder_name"
    mkdir -p "$combined_folder"

    # Now copy files from all a1/a2/a3 subfolders to the corresponding combined folder
    for folder in "$parent_folder"/*; do
        if [ -d "$folder/$subfolder_name" ]; then
            echo "Copying files from $folder/$subfolder_name to $combined_folder"
            cp -r "$folder/$subfolder_name"/* "$combined_folder"
        fi
    done
done

echo "All subfolder contents have been combined."
