#!/bin/bash

# Script to clean assembly files:
# 1. Remove /*alphanum*/ patterns (hex addresses in comments)
# 2. Replace leading whitespace with exactly 8 spaces ONLY on lines that contain those patterns

# Check if file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    echo "If output_file is not specified, input file will be modified in place"
    exit 1
fi

input_file="$1"
output_file="${2:-$input_file}"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist"
    exit 1
fi

# Create temporary file for processing
temp_file=$(mktemp)

# Process the file:
# For lines containing /*alphanum*/ patterns:
# 1. Remove the /*alphanum*/ patterns 
# 2. Replace leading whitespace with exactly 8 spaces
# For other lines: leave unchanged
sed -E '
    # Only process lines that contain /*[alphanum]*/ patterns
    /\/\*[[:alnum:]]+\*\// {
        # Remove /*alphanum*/ patterns (hex addresses and similar)
        s|/\*[[:alnum:]]+\*/||g
        # Replace leading whitespace with exactly 8 spaces
        s|^[[:space:]]*|        |
    }
' "$input_file" > "$temp_file"

# Move temp file to output location
mv "$temp_file" "$output_file"

echo "File processed successfully. Output saved to: $output_file" 