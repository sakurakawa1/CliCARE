#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import os

if len(sys.argv) != 2:
    sys.stderr.write("Usage: python tag_writer.py <step_name>\n")
    sys.exit(1)

step_name = sys.argv[1]
tag = f"\n====Step_{step_name}====\n"

# root dir
BASE_OUT_DIR = "output"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# Read the output from the mapper line by line: patient_no, inpatient_no, record
reader = csv.reader(sys.stdin, delimiter='\t')

# Track the file path of the file for which a label has been written
tagged_files = set()

for line_no, parts in enumerate(reader, start=1):
    if len(parts) != 3:
        sys.stderr.write(f"[WARN] Skipping malformed line {line_no}: {parts}\n")
        continue

    patient, inp, record = parts
    patient = patient.strip()
    inp = inp.strip()
    record = record.strip()

    # Create a directory for the patient
    patient_dir = os.path.join(BASE_OUT_DIR, patient)
    os.makedirs(patient_dir, exist_ok=True)

    out_path = os.path.join(patient_dir, f"{inp}.txt")

    try:
        with open(out_path, "a+", encoding="utf-8") as fout:
            if out_path not in tagged_files:
                fout.seek(0)
                content = fout.read()
                if tag not in content:
                    fout.write(tag)
                tagged_files.add(out_path)
            fout.write(record + "\n")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed writing to {out_path}: {e}\n")
