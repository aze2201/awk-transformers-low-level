#!/usr/bin/env bash

# This script now calls the awk interpreter and tells it to run
# the code saved in the 'transformer.awk' file.

# -f transformer.awk: Specifies the awk script file.
# -v OFS='\t': Sets the Output Field Separator variable.
# -v mode="$1": Passes the first command-line argument (e.g., "train") to the awk script.
# /dev/null: A dummy input file, since the awk script reads from "input.txt" internally.

awk -f transformer.awk -v OFS='\t' -v mode="$1" /dev/null
