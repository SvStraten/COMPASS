import time
import random as rn
import numpy as np
from Setup.transform import transform_data


def _to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        if y.shape[1] == 1:
            return y[:, 0].astype(np.int64)
        return np.argmax(y, axis=1).astype(np.int64)
    raise ValueError(f"Unexpected label shape {y.shape}; expected 1D or 2D.")


class Sampler:
    def __init__(
        self,
        data=0,
        ntasks=2,
        ntrain=4000,
        ntest=500,
    ):
        rn.seed(512)
        _ = time.time() 

        ntrain = len(data.train.get_data())
        ntest = len(data.test_orig.get_data())
        
        feature_attrs = [
            a
            for a in data.logfile.attributes()
            if a != data.logfile.time and a != data.logfile.trace
        ]
        x, y, vals = transform_data(data.logfile, feature_attrs)

        y_classes = _to_class_indices(y)

        n_events = len(data.logfile.get_data())
        samples = np.stack([[x_j[i] for x_j in x] for i in range(n_events)], axis=0)

        tasks = {}
        for q in range(ntasks):
            if q == 0:      
                tasks[q] = samples[:150]
            elif q == 1:   
                tasks[q] = samples[150:300]

        inputs = {}
        labels = {}
        for q in range(ntasks):
            inputs[q] = samples[:ntrain]
            labels[q] = y_classes[:ntrain]

        test_inputs = {}
        test_labels = {}
        for q in range(ntasks):
            test_inputs[q] = samples[-ntest:]
            test_labels[q] = y_classes[-ntest:]

        self.tasks = tasks
        self.inputs = inputs
        self.labels = labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels