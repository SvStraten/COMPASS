import numpy as np
import pandas as pd
from typing import List
import torch

def transform_data(log, columns: List[str]):
    num_activities = len(log.values[log.activity]) + 1

    col_num_vals = {}
    for col in columns:
        if col == log.activity:
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = int(log.contextdata[col].max()) + 2

    inputs: List[List[int]] = []
    n_inputs = len(columns) * log.k - len(log.ignoreHistoryAttributes) * log.k
    for _ in range(n_inputs):
        inputs.append([])

    outputs: List[int] = []
    for _, row in log.contextdata.iterrows():
        i = 0
        for attr in columns:
            if attr not in log.ignoreHistoryAttributes:
                for k in range(log.k):
                    inputs[i].append(int(row[f"{attr}_Prev{k}"]))
                    i += 1
        outputs.append(int(row[log.activity]))
        
    for j in range(len(inputs)):
        inputs[j] = np.asarray(inputs[j], dtype=np.int64)

    outputs = np.asarray(outputs, dtype=np.int64)
    
    return inputs, outputs, col_num_vals
