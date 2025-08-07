#!/usr/bin/env bash


file=$1

grep -E '^\s+[A-Z][A-Z0-9_.]*\s' "$file" | \
    sed -E 's/^\s+([A-Z][A-Z0-9_.]*).*/\1/' | \
    sort | \
    uniq -c | \
    sort -nr