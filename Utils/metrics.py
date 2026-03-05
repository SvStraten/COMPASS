import os
import csv
import numpy as np
from sklearn.metrics import accuracy_score

def save_granular_accuracy(
    dataset_name, 
    method_name, 
    seed, 
    true_labels, 
    pred_labels, 
    output_file="OutputLogs/granular_accuracy.csv", 
    bin_size=500
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Dataset', 'Method', 'Seed', 'Event_Index', 'Accuracy'])
        
        total_events = len(true_labels)
        n_bins = int(np.ceil(total_events / bin_size))
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, total_events)
            
            y_true_bin = true_labels[start_idx:end_idx]
            y_pred_bin = pred_labels[start_idx:end_idx]

            if len(y_true_bin) > 0:
                acc = accuracy_score(y_true_bin, y_pred_bin)
                
                mid_point = start_idx + (len(y_true_bin) // 2)
                
                writer.writerow([dataset_name, method_name, seed, mid_point, acc])

    print(f"Saved granular accuracy to {output_file}")