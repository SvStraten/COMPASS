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

    if os.path.exists(save_path):
        print(f"Loading cached preprocessing for {dataName} from {save_path}")
        with open(save_path, "rb") as f:
            objs = pickle.load(f)
        
        ds = objs["data_sampler"]
        if hasattr(ds, "case_ids"):
            return objs["dataName"], ds
        else:
            print("Cache found but missing 'case_ids'. Re-running preprocessing...")
    
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
        d.prepare(STANDARD)
    
    if hasattr(d, 'test_logfile') and d.test_logfile is not None:
        target_logfile = d.test_logfile
    else:
        target_logfile = d.logfile

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
            break
        except Exception:
            parsed = None
    
    if parsed is None:
        parsed = pd.to_datetime(df["completeTime"], errors="coerce")
    df["completeTime"] = parsed

    if target_logfile.contextdata is None:
        target_logfile.data = df
    else:
        target_logfile.contextdata = df

    data_sampler = Sampler(data=d)

    trace_col_name = "case"
    if hasattr(d.logfile, "trace"): trace_col_name = d.logfile.trace
    elif hasattr(d.logfile, "trace_attr"): trace_col_name = d.logfile.trace_attr
    
    try:
        raw_case_ids = df[trace_col_name].values
    except KeyError:
        raw_case_ids = df.iloc[:, 0].values

    sampler_len = 0
    if isinstance(data_sampler.test_inputs, list) and len(data_sampler.test_inputs) > 0:
        sampler_len = len(data_sampler.test_inputs[0])
    elif isinstance(data_sampler.test_inputs, dict):
        first_key = sorted(data_sampler.test_inputs.keys())[0]
        sampler_len = len(data_sampler.test_inputs[first_key])
    else:
        sampler_len = len(data_sampler.test_inputs)

    if len(raw_case_ids) > sampler_len:
        diff = len(raw_case_ids) - sampler_len
        data_sampler.case_ids = raw_case_ids[-sampler_len:]
    elif len(raw_case_ids) < sampler_len:
        data_sampler.case_ids = raw_case_ids
    else:
        data_sampler.case_ids = raw_case_ids

    with open(save_path, "wb") as f:
        pickle.dump({"dataName": dataName, "data_sampler": data_sampler}, f)
        
    print(f"Preprocessed objects saved to {save_path}")
    return dataName, data_sampler