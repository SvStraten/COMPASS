import os
import pickle
import pandas as pd
import numpy as np
import sys
from typing import Any, Dict, List
from Data.data import Data
from Setup.LogFile import LogFile
from Setup.transform import transform_data
from Setup.setting import Setting
from Setup.sampler import Sampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STANDARD = Setting(10, "train-test", True, False, 0, 10) if 'Setting' in locals() else None

def preprocess(file: str, save_dir: str = "Preprocessed"):
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(file)
    dataName = os.path.splitext(filename)[0]
    
    save_path = os.path.join(save_dir, f"{dataName}_preprocessed.pkl")

    # 1. Load cache if it exists
    if os.path.exists(save_path):
        print(f"Loading cached preprocessing for {dataName} from {save_path}")
        with open(save_path, "rb") as f:
            objs = pickle.load(f)
        
        ds = objs["data_sampler"]
        if hasattr(ds, "case_ids"):
            return objs["dataName"], ds
        else:
            print("Cache found but missing 'case_ids'. Re-running preprocessing...")

    # 2. Fresh preprocess
    print(f"\n--- START PREPROCESSING: {dataName} ---")
    
    if Data is None:
        raise ImportError("The 'Data' or 'Setup' modules are missing from the root directory.")

    try:
        _ = pd.read_csv(file, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {file}")
        
    print("Dataset:", dataName)

    d = Data(
        dataName,
        LogFile(
            filename=file,
            delim=",",
            header=0,
            rows=None,
            time_attr="completeTime",
            trace_attr="case",
            activity_attr="event",
            convert=False, 
        ),
    )

    d.logfile.keep_attributes(["event", "completeTime"])
    
    if STANDARD:
        print(f"Applying STANDARD settings (Split={STANDARD.train_percentage})...")
        d.prepare(STANDARD)
    
    print("### Data Prepared")
    
    if hasattr(d, 'test_logfile') and d.test_logfile is not None:
        target_logfile = d.test_logfile
        print(f"DEBUG: Using TEST LogFile. Rows: {len(target_logfile.data)}")
    else:
        target_logfile = d.logfile
        print(f"DEBUG: Using FULL LogFile. Rows: {len(target_logfile.data)}")

    df = target_logfile.get_data().copy()
    
    parsed = None
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
    ):
        try:
            parsed = pd.to_datetime(df["completeTime"], format=fmt, exact=True)
            print("Detected time format:", fmt)
            break
        except Exception:
            parsed = None
    
    if parsed is None:
        parsed = pd.to_datetime(df["completeTime"], errors="coerce")
        print("Time format: auto-inferred (coerce).")
    df["completeTime"] = parsed

    if target_logfile.contextdata is None:
        target_logfile.data = df
    else:
        target_logfile.contextdata = df

    print("Initializing Sampler...")
    data_sampler = Sampler(data=d)

    trace_col_name = "case"
    if hasattr(d.logfile, "trace"): trace_col_name = d.logfile.trace
    elif hasattr(d.logfile, "trace_attr"): trace_col_name = d.logfile.trace_attr
    
    try:
        raw_case_ids = df[trace_col_name].values
        print(f"DEBUG: Extracted {len(raw_case_ids)} raw case IDs from column '{trace_col_name}'.")
    except KeyError:
        print(f"WARNING: Column '{trace_col_name}' not found. Using first column.")
        raw_case_ids = df.iloc[:, 0].values

    sampler_len = 0
    if isinstance(data_sampler.test_inputs, list) and len(data_sampler.test_inputs) > 0:
        sampler_len = len(data_sampler.test_inputs[0])
    elif isinstance(data_sampler.test_inputs, dict):
        first_key = sorted(data_sampler.test_inputs.keys())[0]
        sampler_len = len(data_sampler.test_inputs[first_key])
    else:
        sampler_len = len(data_sampler.test_inputs)

    print(f"DEBUG: Sampler contains {sampler_len} sequences.")

    if len(raw_case_ids) > sampler_len:
        diff = len(raw_case_ids) - sampler_len
        print(f"[ALIGNMENT] Mismatch detected: Raw Log={len(raw_case_ids)}, Sampler={sampler_len}")
        print(f"[ALIGNMENT] Trimming {diff} rows from the BEGINNING of Case IDs.")
        data_sampler.case_ids = raw_case_ids[-sampler_len:]
    elif len(raw_case_ids) < sampler_len:
        print(f"CRITICAL WARNING: Sampler has MORE data ({sampler_len}) than Raw Log ({len(raw_case_ids)}).")
        data_sampler.case_ids = raw_case_ids
    else:
        print(f"[ALIGNMENT] Perfect match ({sampler_len} rows).")
        data_sampler.case_ids = raw_case_ids

    with open(save_path, "wb") as f:
        pickle.dump({"dataName": dataName, "data_sampler": data_sampler}, f)
        
    print(f"Preprocessed objects saved to {save_path}")
    print("--- PREPROCESSING COMPLETE ---\n")
    return dataName, data_sampler