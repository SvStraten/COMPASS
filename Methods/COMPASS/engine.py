import os
import csv
import time
import collections
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _cpu_ram_mb() -> float:
    if _PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return float("nan")


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


class COMPASSEngine:
    DRIFT_DATA = {
        "IRO5000":                   [2456, 7664, 13164, 18308, 23869, 29031, 34656, 39838, 45354],
        "ORI5000":                   [2325, 7526, 13276, 18426, 24280, 29448, 35083, 40271, 45941],
        "ROI5000":                   [2633, 7834, 13019, 18167, 23471, 28633, 33834, 39016, 44266],
        "OIR5000":                   [3273, 8475, 14405, 19554, 25414, 30581, 36496, 41683, 47562],
        "RIO5000":                   [2903, 8108, 13643, 18793, 24345, 29513, 35005, 40193, 45658],
        "BPIC2015_Recurrent":        [4006, 22338, 38097, 51712, 70044, 85803, 99418],
        "RequestForPayment":         [289, 3674],
        "InternationalDeclarations": [2905, 7799],
        "DomesticDeclarations":      [1089, 5712],
    }

    def __init__(
        self,
        model: torch.nn.Module,
        keeplora_handler,
        device: Optional[str] = None,
        window_size: Optional[int] = None,
        epochs_per_window: Optional[int] = None,
        learning_rate: float = 2e-4,
        input_dtype: str = "long",
        verbose: bool = True,
        make_plots: bool = False,
        plot_dir: str = "runs",
        data_name: Optional[str] = None,
        model_name: Optional[str] = None,
        ntasks: Optional[int] = 1,
        hard_buffer_size: int = 0,
        batch_size: Optional[int] = None,
        svd_limit_batches: int = 20,
        loss_window_length: int = 5,
        loss_variance_threshold: float = 0.0005,
        svd_cooldown_windows: int = 10,
        use_svd: bool = True,
        use_oracle_drift: bool = False,
        save_per_event: bool = True,
    ):
        self.model = model
        self.keeplora_handler = keeplora_handler
        self.device = device
        self.model.to(self.device)
        self.recent_buffer_size = window_size
        self.ntasks = ntasks or 1
        self.input_dtype = input_dtype
        self.verbose = verbose
        self.plot_dir = plot_dir
        self.data_name = data_name
        self.gradient_steps = epochs_per_window
        self.hard_buffer_size = hard_buffer_size
        self.hard_buffer_inputs = []
        self.hard_buffer_labels = []
        self.batch_size = batch_size
        self.use_svd = use_svd
        self.use_oracle_drift = use_oracle_drift
        self.save_per_event = save_per_event

        self.model_tag = (model_name or "unknown").split("/")[-1]
        self.protocol  = "oracle" if use_oracle_drift else "taskfree"

        self.oracle_drift_indices = []
        if self.use_oracle_drift and self.data_name:
            clean_name = self.data_name.replace(".csv", "").replace("Data/", "")
            for key in self.DRIFT_DATA:
                if key in clean_name:
                    self.oracle_drift_indices = sorted(self.DRIFT_DATA[key])
                    break
            if self.verbose:
                print(f"Oracle: {len(self.oracle_drift_indices)} drift points for {clean_name}")

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.plateau_maxlen = loss_window_length
        self.plateau_buffer = collections.deque(maxlen=loss_window_length)
        self.loss_variance_threshold = loss_variance_threshold
        self.svd_cooldown_windows = svd_cooldown_windows
        self.current_cooldown = 0
        self.plateau_triggers = 0

        self.keeplora_handler.ensure_only_B_trainable(self.model)
        self._rebuild_optimizer(learning_rate)
        self.learning_rate = learning_rate

        if self.verbose:
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable params: {n_trainable:,} | use_svd={use_svd} | oracle={use_oracle_drift}")

    def _rebuild_optimizer(self, lr=None):
        lr = lr or self.learning_rate
        self.optimizer = Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def _to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return (x.long() if self.input_dtype == "long" else x.float()).to(self.device)

    def _to_labels(self, y):
        if isinstance(y, np.ndarray): yt = torch.from_numpy(y)
        elif torch.is_tensor(y): yt = y
        else: yt = torch.tensor(y)
        return yt.long().to(self.device)

    def _update_hard_buffer(self, x_batch, y_batch):
        if self.hard_buffer_size <= 0:
            return
        self.model.eval()
        with torch.no_grad():
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")(
                self.model(self._to_tensor(x_batch)), self._to_labels(y_batch)
            ).cpu().numpy()
        merged = [(float(lb), xb, yb) for xb, yb, lb in zip(x_batch, y_batch, per_sample_loss)]
        merged += [(0.0, xb, yb) for xb, yb in zip(self.hard_buffer_inputs, self.hard_buffer_labels)]
        merged.sort(key=lambda t: t[0], reverse=True)
        merged = merged[:self.hard_buffer_size]
        self.hard_buffer_inputs = [m[1] for m in merged]
        self.hard_buffer_labels = [m[2] for m in merged]

    def _get_hard_buffer_batch(self):
        if not self.hard_buffer_inputs:
            return None, None
        return np.stack(self.hard_buffer_inputs, axis=0), np.asarray(self.hard_buffer_labels)

    def _consolidate_task(self, x_win, y_win, reason="Plateau"):
        if not self.use_svd:
            return 0.0
        self.plateau_triggers += 1
        print(f"\n{reason} — consolidating task #{self.plateau_triggers}")
        t0 = time.perf_counter()
        self.keeplora_handler.consolidate_task(
            model=self.model,
            x_tensor=self._to_tensor(x_win),
            y_tensor=self._to_labels(y_win),
            criterion=self.criterion,
        )
        elapsed = time.perf_counter() - t0
        self._rebuild_optimizer()
        return elapsed

    def _check_and_handle_plateau(self, loss, x_win, y_win):
        if self.use_oracle_drift or not self.use_svd:
            return 0.0, float("nan")
        self.plateau_buffer.append(loss)
        if len(self.plateau_buffer) < self.plateau_maxlen:
            return 0.0, float("nan")
        variance = float(np.var(np.array(self.plateau_buffer)))
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return 0.0, variance
        if variance < self.loss_variance_threshold:
            elapsed = self._consolidate_task(x_win, y_win, reason="Low Variance")
            self.plateau_buffer.clear()
            self.current_cooldown = self.svd_cooldown_windows
            return elapsed, variance
        return 0.0, variance

    def run_stream(self, data):
        return self.method(data)

    def method(self, data):
        all_preds, all_true, window_accuracies = [], [], []
        total_correct = total_seen = 0
        self.plateau_triggers = 0
        global_idx = 0
        self.model.train()
        per_event_records = []
        window_records = []
        print("Starting stream...")

        # Flatten stream
        if isinstance(data.test_inputs, dict):
            task_ids = sorted(data.test_inputs.keys())
            input_stream = np.concatenate([data.test_inputs[k] for k in task_ids])
            label_stream = np.concatenate([data.test_labels[k] for k in task_ids])
        elif isinstance(data.test_inputs, list):
            if len(data.test_inputs) > 0 and isinstance(data.test_inputs[0], (list, np.ndarray)):
                input_stream = np.concatenate(data.test_inputs)
                label_stream = np.concatenate(data.test_labels)
            else:
                input_stream = np.array(data.test_inputs)
                label_stream = np.array(data.test_labels)
        else:
            input_stream = data.test_inputs
            label_stream = data.test_labels

        W = self.recent_buffer_size
        N = len(input_stream)
        n_full = (N + W - 1) // W

        for i in range(n_full):
            s_win = i * W
            e_win = min(s_win + W, N)
            x_win = input_stream[s_win:e_win]
            y_win = label_stream[s_win:e_win]
            win_size = len(y_win)

            consolidation_time_s = 0.0
            triggered_consolidation = False
            if self.use_oracle_drift and self.use_svd:
                win_end = global_idx + win_size
                for dp in self.oracle_drift_indices:
                    if global_idx < dp <= win_end:
                        consolidation_time_s = self._consolidate_task(
                            x_win, y_win, reason=f"Oracle@{dp}"
                        )
                        triggered_consolidation = True
                        break
               
            #inference
            self.model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                probs = F.softmax(self.model(self._to_tensor(x_win)), dim=1).cpu().numpy()
                y_hat = np.argmax(probs, axis=1)
            inf_time_per_event = (time.perf_counter() - t0) / win_size

            total_correct += int((y_hat == y_win).sum())
            total_seen    += win_size
            all_preds.extend(y_hat)
            all_true.extend(y_win)
            test_acc = float((y_hat == y_win).mean())
            window_accuracies.append(test_acc)
            gpu_mem = _gpu_mem_mb()
            cpu_mem = _cpu_ram_mb()

            #training
            self.model.train()
            self._update_hard_buffer(x_win, y_win)
            cur_n = len(y_win)
            Xb, yb = self._get_hard_buffer_batch()
            buf_n  = len(yb) if Xb is not None else 0
            accumulated_loss = 0.0

            t0 = time.perf_counter()
            for step in range(self.gradient_steps):
                self.optimizer.zero_grad(set_to_none=True)
                total_samples = cur_n + buf_n
                if total_samples == 0:
                    continue
                sum_loss = torch.tensor(0.0, device=self.device)

                def fwd_bwd(x_data, y_data, n):
                    local = torch.tensor(0.0, device=self.device)
                    bs = self.batch_size or n
                    for s in range(0, n, bs):
                        bx = self._to_tensor(x_data[s:s+bs])
                        by = self._to_labels(y_data[s:s+bs])
                        l = self.criterion(self.model(bx), by)
                        local += l.detach()
                        (l / float(total_samples)).backward()
                    return local

                sum_loss += fwd_bwd(x_win, y_win, cur_n)
                if buf_n > 0:
                    sum_loss += fwd_bwd(Xb, yb, buf_n)
                self.optimizer.step()
                accumulated_loss += (sum_loss / float(total_samples)).item()

            train_time_per_event = (time.perf_counter() - t0) / win_size
            avg_loss = accumulated_loss / self.gradient_steps

            #consolidation
            plateau_time, loss_variance = self._check_and_handle_plateau(avg_loss, x_win, y_win)
            if plateau_time > 0.0:
                consolidation_time_s += plateau_time
                triggered_consolidation = True

            consolidation_time_per_event = consolidation_time_s / win_size
            cumulative_acc = total_correct / total_seen

            if self.verbose:
                print(f"Window {i+1}/{n_full} | Loss: {avg_loss:.4f} | Acc: {test_acc:.4f}")
                
            if self.save_per_event:
                for j in range(win_size):
                    p = probs[j]
                    sorted_p = np.sort(p)[::-1]
                    per_event_records.append({
                        "event_idx":               global_idx + j,
                        "window_idx":              i,
                        "true_label":              int(y_win[j]),
                        "pred_label":              int(y_hat[j]),
                        "correct":                 int(y_hat[j] == y_win[j]),
                        "confidence":              f"{float(p.max()):.6f}",
                        "entropy":                 f"{float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))):.6f}",
                        "margin":                  f"{float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0:.6f}",
                        "window_loss":             f"{avg_loss:.6f}",
                        "window_accuracy":         f"{test_acc:.6f}",
                        "cumulative_accuracy":     f"{cumulative_acc:.6f}",
                        "inference_time_s":        f"{inf_time_per_event:.6f}",
                        "train_time_s":            f"{train_time_per_event:.6f}",
                        "consolidation_time_s":    f"{consolidation_time_per_event:.6f}",
                        "total_event_time_s":      f"{inf_time_per_event + train_time_per_event + consolidation_time_per_event:.6f}",
                        "gpu_mem_mb":              f"{gpu_mem:.2f}",
                        "cpu_ram_mb":              f"{cpu_mem:.2f}",
                        "consolidation_triggered": int(triggered_consolidation),
                        "consolidation_count":     self.plateau_triggers,
                    })

            window_records.append({
                "window_idx":          i,
                "window_start_event":  s_win,
                "window_end_event":    e_win - 1,
                "avg_loss":            f"{avg_loss:.6f}",
                "loss_variance":       f"{loss_variance:.8e}" if not np.isnan(loss_variance) else "nan",
                "drift_detected":      int(triggered_consolidation),
                "consolidation_count": self.plateau_triggers,
            })

            global_idx += win_size

        if self.save_per_event and per_event_records:
            os.makedirs(self.plot_dir, exist_ok=True)
            fname = f"COMPASS_{self.data_name}_{self.model_tag}_{self.protocol}_per_event.csv"
            path  = os.path.join(self.plot_dir, fname)
            with open(path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_event_records[0].keys()))
                writer.writeheader()
                writer.writerows(per_event_records)
            print(f"[COMPASS] Per-event CSV → {path}")

        if window_records:
            os.makedirs(self.plot_dir, exist_ok=True)
            tau_str  = f"{self.loss_variance_threshold:.0e}"
            fname_w  = (
                f"COMPASS_{self.data_name}_{self.model_tag}_{self.protocol}"
                f"_W{W}_tau{tau_str}_drift_windows.csv"
            )
            path_w = os.path.join(self.plot_dir, fname_w)
            with open(path_w, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(window_records[0].keys()))
                writer.writeheader()
                writer.writerows(window_records)

        total_acc   = total_correct / total_seen if total_seen > 0 else 0.0
        macro_f1    = f1_score(all_true, all_preds, average="macro",    zero_division=0)
        weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
        print(f"\nAcc: {total_acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | Consolidations: {self.plateau_triggers}")

        return {
            "prediction_results": {"actual_labels": all_true, "prediction_labels": all_preds},
            "window_accuracies":  window_accuracies,
            "overall_accuracy":   total_acc,
            "macro_f1":           macro_f1,
            "weighted_f1":        weighted_f1,
            "plateau_triggers":   self.plateau_triggers,
        }