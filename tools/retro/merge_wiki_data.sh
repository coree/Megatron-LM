# Set the directory containing the wiki files
WIKI_DIR="/home/joseph.cornelius/projects/Megatron-LM/data/enwiki-latest-pages-articles"
# Set the output directory where the consolidated file will be saved
OUTDIR="/home/joseph.cornelius/projects/Megatron-LM/data/enwiki-latest-single-file"

# Create the output directory if it does not exist
mkdir -p $OUTDIR

# Remove the existing consolidated file if it exists
rm $OUTDIR/wiki_all.json

# Create a new empty consolidated file
touch $OUTDIR/wiki_all.json

# Find all files in the WIKI_DIR and its subdirectories
# -print0 ensures file names with special characters are handled correctly
find "$WIKI_DIR" -type f  -print0 |
    # Read each file path found by 'find'
    while IFS= read -r -d '' line; do
            # Extract the filename from the full path
            filename=$(echo "$line" | rev | cut -d'/' -f 1 | rev)
            # Extract the name of the subdirectory containing the file
            subfilename=$(echo "$line" | rev | cut -d'/' -f 2 | rev)
            # Create a prefix combining the subdirectory name and filename
            prefix="${subfilename}_${filename}"
            # Store the full path in 'new_name'
            new_name=$(echo "$line")
            # Log the processing information
            echo "Procesing $prefix, $filename, $new_name"
            # Append the content of the current file to the consolidated file
            cat $new_name >> $OUTDIR/wiki_all.json
    done