#!/usr/bin/env bash

kernel="${1}"
compute-sanitizer --tool racecheck tools/debug/sanity_check.py --kernel=${kernel} --small > debug/sanitize.txt