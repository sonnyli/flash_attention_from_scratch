#!/usr/bin/env bash

# Exit on any error
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
filename=$TIMESTAMP
if [ ! -z "$1" ]; then
    filename=$1
fi

mkdir -p "logs" "regs"
# Create a temporary file. The XXXXXX will be replaced by random alphanumeric characters.
TMPFILE="./logs/$filename"
rm -f "$TMPFILE" "$TMPFILE.demangled"

echo "Using temporary log file: $TMPFILE"

# Run pip install, logging output to the temporary file
if ! TORCH_CUDA_ARCH_LIST="8.0" PYTHONWARNINGS="ignore" pip install --no-build-isolation . --log "$TMPFILE"; then
    echo "ERROR: pip install failed. Printing log:"
    cat "$TMPFILE"
    exit 1
fi

cat "$TMPFILE" | cu++filt -p > "$TMPFILE.demangled"

echo
echo
# On successful install, run the parse script
python ./tools/build/parse_ptx_build.py "$TMPFILE.demangled" --csv --output "./regs/${filename}_regs.csv"