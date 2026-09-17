#!/bin/bash

ALIGN_SIZE=64

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_file]" >&2
    exit 1
fi

INPUT_FILE="$1"

if [ -n "${2:-}" ]; then
    OUTPUT_FILE="$2"
else
    INPUT_BASENAME=$(basename -- "$INPUT_FILE")
    INPUT_DIR=$(dirname -- "$INPUT_FILE")

    if [[ "$INPUT_BASENAME" == *.* ]]; then
        OUTPUT_FILE="${INPUT_DIR}/${INPUT_BASENAME%.*}_padded.${INPUT_BASENAME##*.}"
    else
        OUTPUT_FILE="${INPUT_DIR}/${INPUT_BASENAME}_padded"
    fi
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE" >&2
    exit 1
fi

FILE_SIZE=$(stat -c%s "$INPUT_FILE")
PADDING_SIZE=$(( (ALIGN_SIZE - (FILE_SIZE % ALIGN_SIZE)) % ALIGN_SIZE ))

cp "$INPUT_FILE" "$OUTPUT_FILE"

if [ "$PADDING_SIZE" -gt 0 ]; then
    dd if=/dev/zero bs=1 count="$PADDING_SIZE" >> "$OUTPUT_FILE" 2>/dev/null
fi

FINAL_SIZE=$(stat -c%s "$OUTPUT_FILE")
OUTPUT_VALUE=$(( FINAL_SIZE / ALIGN_SIZE - 1 ))

echo "File padded: $OUTPUT_FILE (original: ${FILE_SIZE}B, padding: ${PADDING_SIZE}B, final: ${FINAL_SIZE}B)" >&2
echo "$OUTPUT_VALUE"
