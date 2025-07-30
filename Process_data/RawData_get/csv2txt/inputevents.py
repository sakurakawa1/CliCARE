#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

# （inpatient_no -> patient_no）
relation = {}
try:
    with open("patient_admissions.txt", "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                sys.stderr.write(f"[WARN] Line {line_no} format error: {line}\n")
                continue

            patient = parts[0].strip()
            inpatients = parts[1].strip().split(',')

            for inpatient in inpatients:
                inpatient = inpatient.strip()
                if not inpatient:
                    continue
                relation[inpatient] = patient
                
                # sys.stderr.write(f"[INFO] Loaded: {patient},{inpatient}\n")

except Exception as e:
    sys.stderr.write(f"[ERROR] Failed to read patient_inpatient.txt: {e}\n")
    sys.exit(1)

FIELDS = ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'amount', 'amountuom', 'rate', 'rateuom', 'patientweight']

# read CSV
try:
    with open("/Cancer_MIMIC-IV/cancer_hosp/inputevents.csv", "r", newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, quotechar='"', skipinitialspace=True)
        header = next(reader)
        header = [h.strip() for h in header]
        idx = {name: i for i, name in enumerate(header)}

        missing_fields = [f for f in FIELDS if f not in idx]
        if missing_fields:
            raise ValueError(f"[ERROR] Missing fields: {missing_fields}")

        for line_no, row in enumerate(reader, start=2):
            try:
                inp = row[idx["hadm_id"]].strip()
                patient = relation.get(inp)
                if not patient:
                    continue
                parts = []
                for fld in FIELDS:
                    val = row[idx[fld]].strip().replace('\n', ' ').replace('\r', ' ') if idx[fld] < len(row) else ""
                    parts.append(f"{fld}:{val}")
                record = ",".join(parts) + ";"
                print(f"{patient}\t{inp}\t{record}")
            except Exception as e:
                sys.stderr.write(f"[EXCEPTION] Line {line_no} error: {e} | Row: {row}\n")

except Exception as e:
    sys.stderr.write(f"[ERROR] Failed to process CSV: {e}\n")