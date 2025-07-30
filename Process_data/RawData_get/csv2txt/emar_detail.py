#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
# 1. inpatient_no -> patient_no
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
                if inpatient:
                    relation[inpatient] = patient
                    # sys.stderr.write(f"[INFO] Loaded: {patient},{inpatient}\n")
except Exception as e:
    sys.stderr.write(f"[ERROR] Failed to read patient_inpatient.txt: {e}\n")
    sys.exit(1)

# 2. read CDR_LIS_MAIN_bak.csv，REPORTID -> PATSERIALNO
reportid_to_inp = {}
try:
    with open("/LLM_DATA_PROCESS/emartest.csv", "r", newline='', encoding="utf-8") as f_main:
        reader_main = csv.reader(f_main, quotechar='"', skipinitialspace=True)
        header_main = next(reader_main)
        idx_main = {name.strip(): i for i, name in enumerate(header_main)}

        for row in reader_main:
            reportid = row[idx_main["emar_id"]].strip()
            inpatient= row[idx_main["hadm_id"]].strip().split('.')[0]# inp = row[idx["hadm_id"]].strip().split('.')[0]
            reportid_to_inp[reportid] = inpatient
except Exception as e:
    sys.stderr.write(f"[ERROR] Failed to read CDR_LIS_MAIN_bak.csv: {e}\n")
    sys.exit(1)

# 3. read CDR_LIS_BIOCHEMISTRY_RESULT.csv to get join，print target information
FIELDS = ['subject_id', 'emar_id', 'emar_seq', 'dose_given', 'dose_given_unit', 'product_description', 'infusion_rate', 'infusion_rate_unit', 'route']
try:
    with open("/Cancer_MIMIC-IV/cancer_hosp/emar_detail.csv", "r", newline='', encoding="utf-8") as f_bio:
        reader_bio = csv.reader(f_bio, quotechar='"', skipinitialspace=True)
        header_bio = next(reader_bio)
        header_bio = [h.strip() for h in header_bio]
        idx_bio = {name: i for i, name in enumerate(header_bio)}

        for line_no, row in enumerate(reader_bio, start=2):
            try:
                reportid = row[idx_bio["emar_id"]].strip()
                inpatient = reportid_to_inp.get(reportid)
                if not inpatient:
                    continue
                patient = relation.get(inpatient)
                if not patient:
                    continue

                # extraction
                parts = []
                selected_fields = FIELDS if FIELDS else header_bio
                for fld in selected_fields:
                    val = row[idx_bio[fld]].strip().replace("\n", " ").replace("\r", " ") if fld in idx_bio else ""
                    parts.append(f"{fld}:{val}")

                record = ",".join(parts) + ";"
                print(f"{patient}\t{inpatient}\t{record}")

            except Exception as e:
                sys.stderr.write(f"[EXCEPTION] Line {line_no} error: {e} | Row: {row}\n")
except Exception as e:
    sys.stderr.write(f"[ERROR] Failed to read CDR_LIS_BIOCHEMISTRY_RESULT.csv: {e}\n")
