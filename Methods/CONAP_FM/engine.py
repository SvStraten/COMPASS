import collections
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

class OnlineCaLoraEngine:

    DRIFT_DATA = {
        "IRO5000": [2456, 7664, 13164, 18308, 23869, 29031, 34656, 39838, 45354],
        "ORI5000": [2325, 7526, 13276, 18426, 24280, 29448, 35083, 40271, 45941],
        "ROI5000": [2633, 7834, 13019, 18167, 23471, 28633, 33834, 39016, 44266],
        "OIR5000": [3273, 8475, 14405, 19554, 25414, 30581, 36496, 41683, 47562],
        "RIO5000": [2903, 8108, 13643, 18793, 24345, 29513, 35005, 40193, 45658],
        "BPIC2015_Recurrent": [4006, 22338, 38097, 51712, 70044, 85803, 99418],
        "RequestForPayment": [289, 3674],
        "InternationalDeclarations": [2905, 7799],
        "DomesticDeclarations": [1089, 5712]
    }

    def __init__(self,
                 model: torch.nn.Module,
                 calora_handler, 
                 device: Optional[str] = None,
                 window_size: Optional[int] = None,
                 epochs_per_window: Optional[int] = None,
                 learning_rate: float = 2e-3,
                 input_dtype: str = "long",
                 verbose: bool = True,
                 make_plots: bool = False,
                 plot_dir: str = "runs",
                 data_name: Optional[str] = None,
                 ntasks: Optional[int] = 1,
                 hard_buffer_size: int = 4,
                 batch_size: Optional[int] = None,
                 svd_limit_batches: int = 20,
                 loss_window_length: int = 5,        
                 loss_variance_threshold: float = 0.05, 
                 svd_cooldown_windows: int = 10,
                 use_paca: bool = True,
                 use_caga: bool = True,
                 use_svd: bool = True,
                 use_oracle_drift: bool = False
                 ):
        
        self.model = model
        self.calora_handler = calora_handler
        self.device = device
        self.model.to(self.device)
        self.recent_buffer_size = window_size
        self.ntasks = ntasks or 1
        self.input_dtype = input_dtype
        self.verbose = verbose
        self.make_plots = make_plots
        self.plot_dir = plot_dir
        self.data_name = data_name
        self.svd_limit_batches = svd_limit_batches
        self.gradient_steps = epochs_per_window
        self.hard_buffer_size = hard_buffer_size
        self.hard_buffer_inputs = [] 
        self.hard_buffer_labels = [] 
        self.batch_size = batch_size
        self.use_paca = use_paca
        self.use_caga = use_caga
        self.use_svd = use_svd
        self.use_oracle_drift = use_oracle_drift

        self.oracle_drift_indices = []
        if self.use_oracle_drift and self.data_name:
            clean_name = self.data_name.replace(".csv", "").replace("Data/", "")
            for key in self.DRIFT_DATA:
                if key in clean_name:
                    self.oracle_drift_indices = sorted(self.DRIFT_DATA[key])
                    break
            if self.verbose:
                print(f"Loaded {len(self.oracle_drift_indices)} drift points for {clean_name}")

        # Optimization
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(params, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.plateau_maxlen = loss_window_length
        self.plateau_buffer = collections.deque(maxlen=loss_window_length)
        self.loss_variance_threshold = loss_variance_threshold
        
        self.svd_cooldown_windows = svd_cooldown_windows
        self.current_cooldown = 0
        self.plateau_triggers = 0
        
        if self.verbose:
            print(f"PACA={self.use_paca}, CAGA={self.use_caga}, SVD={self.use_svd}, OracleDrift={self.use_oracle_drift}")

    def _to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if self.input_dtype == "long":
            x = x.long()
        else:
            x = x.float()
        return x.to(self.device)

    def _to_labels(self, y):
        if isinstance(y, np.ndarray): yt = torch.from_numpy(y)
        elif torch.is_tensor(y): yt = y
        else: yt = torch.tensor(y)
        return yt.long().to(self.device)

    def _update_hard_buffer(self, x_batch, y_batch):
        if self.hard_buffer_size <= 0: return
        self.model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(x_batch)
            y_t = self._to_labels(y_batch)
            logits = self.model(x_t)
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")(logits, y_t)
            batch_losses = per_sample_loss.cpu().numpy()

        merged = []
        for xb, yb, lb in zip(x_batch, y_batch, batch_losses):
            merged.append((float(lb), xb, yb))
        for xb, yb in zip(self.hard_buffer_inputs, self.hard_buffer_labels):
            merged.append((0.0, xb, yb))

        merged.sort(key=lambda t: t[0], reverse=True)
        merged = merged[: self.hard_buffer_size]
        self.hard_buffer_inputs = [m[1] for m in merged]
        self.hard_buffer_labels = [m[2] for m in merged]

    def _get_hard_buffer_batch(self):
        if not self.hard_buffer_inputs: return None, None
        Xb = np.stack(self.hard_buffer_inputs, axis=0)
        yb = np.asarray(self.hard_buffer_labels)
        return Xb, yb

    def _extract_subspace(self, x_win, y_win, reason="Plateau"):
        if not self.use_svd: return

        self.plateau_triggers += 1
        print(f"\n[Trigger: {reason}] Extracting subspace... (#{self.plateau_triggers})")
        
        X_ten = self._to_tensor(x_win)
        y_ten = self._to_labels(y_win)
        
        dataset = torch.utils.data.TensorDataset(X_ten, y_ten)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size or 32, shuffle=False)
        
        self.model.train() 
        self.model.zero_grad()
        accumulated_grads = {}
        
        batch_count = 0
        for bx, by in loader:
            if batch_count >= self.svd_limit_batches: break
            
            logits = self.model(bx)
            loss = self.criterion(logits, by)
            loss.backward()

            for name, param in self.model.named_parameters():
                if "lora" in name and param.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = param.grad.detach().clone()
                    else:
                        accumulated_grads[name] += param.grad.detach()
            
            self.model.zero_grad()
            batch_count += 1

        new_u_matrices = {}
        for name, grad in accumulated_grads.items():
            grad_f32 = grad.float()
            try:
                U, S, Vh = torch.linalg.svd(grad_f32, full_matrices=False)
                energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
                threshold = self.calora_handler.projection_threshold
                k_indices = (energy >= threshold).nonzero(as_tuple=True)[0]
                k = k_indices[0].item() + 1 if len(k_indices) > 0 else len(S)
                new_u_matrices[name] = Vh[:k, :].T 
            except Exception as e:
                print(f"SVD Failed for {name}: {e}")

        for name, U_new in new_u_matrices.items():
            if name not in self.calora_handler.old_u_matrices:
                self.calora_handler.old_u_matrices[name] = U_new
            else:
                U_old = self.calora_handler.old_u_matrices[name]
                combined = torch.cat([U_old, U_new], dim=1)
                try:
                    Uc, Sc, Vhc = torch.linalg.svd(combined, full_matrices=False)
                    self.calora_handler.old_u_matrices[name] = Uc 
                except:
                    self.calora_handler.old_u_matrices[name] = U_new
        print(f"Subspace updated.\n")

    def _check_and_handle_plateau(self, current_window_loss, x_win, y_win):
        if self.use_oracle_drift or not self.use_svd:
            return

        self.plateau_buffer.append(current_window_loss)

        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return

        if len(self.plateau_buffer) < self.plateau_maxlen:
            return

        losses = np.array(self.plateau_buffer)
        variance = np.var(losses)
        
        if variance < self.loss_variance_threshold:
            self._extract_subspace(x_win, y_win, reason="Low Variance")
            self.plateau_buffer.clear()
            self.current_cooldown = self.svd_cooldown_windows

    def run_stream(self, data):
        return self.method(data)

    def method(self, data):
        all_preds = []
        all_true = []
        window_accuracies = []
        
        total_correct = 0; total_seen = 0
        self.plateau_triggers = 0 
        global_idx = 0

        self.model.train()
        print(f"Starting CONAP-FM...")

        # Flatten Data Stream
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
        
            if self.use_oracle_drift and self.use_svd:
                win_start = global_idx
                win_end = global_idx + len(x_win)
                for dp in self.oracle_drift_indices:
                    if win_start < dp <= win_end:
                        self._extract_subspace(x_win, y_win, reason=f"Oracle Drift @ {dp}")
                        break

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                x_eval = self._to_tensor(x_win)
                logits_eval = self.model(x_eval)
                y_hat = torch.argmax(logits_eval, dim=1).cpu().numpy()
                correct = int((y_hat == y_win).sum())
                total_correct += correct; total_seen += len(y_win)
                all_preds.extend(y_hat)
                all_true.extend(y_win)
            
            test_acc = float((y_hat == y_win).mean())
            window_accuracies.append(test_acc)

            # Training
            self.model.train()
            self._update_hard_buffer(x_win, y_win)
            
            cur_n = len(y_win)
            Xb, yb = self._get_hard_buffer_batch()
            buf_n = len(yb) if Xb is not None else 0
            
            accumulated_window_loss = 0.0

            for step in range(self.gradient_steps):
                self.optimizer.zero_grad(set_to_none=True)
                total_samples = cur_n + buf_n
                if total_samples == 0: continue
                sum_loss = torch.tensor(0.0, device=self.device)

                def forward_backward_microbatches(x_data, y_data, n_samples):
                    local_sum = torch.tensor(0.0, device=self.device)
                    bs = self.batch_size or n_samples
                    for start in range(0, n_samples, bs):
                        end = start + bs
                        bx = self._to_tensor(x_data[start:end])
                        by = self._to_labels(y_data[start:end])
                        logits = self.model(bx)
                        loss_sum = self.criterion(logits, by)
                        local_sum += loss_sum.detach()
                        (loss_sum / float(total_samples)).backward()
                    return local_sum

                sum_loss += forward_backward_microbatches(x_win, y_win, cur_n)
                if buf_n > 0: sum_loss += forward_backward_microbatches(Xb, yb, buf_n)
                
                data_loss = sum_loss / float(total_samples)
                accumulated_window_loss += data_loss.item()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None and "lora" in name:
                            if self.use_paca:
                                paca_weight = self.calora_handler.compute_paca_mask(param, enable=True)
                                param.grad *= paca_weight
                            if self.use_caga and self.calora_handler.old_u_matrices:
                                param.grad = self.calora_handler.apply_caga_correction(name, param, enable=True)

                self.optimizer.step()

            avg_window_loss = accumulated_window_loss / self.gradient_steps
            
            self._check_and_handle_plateau(avg_window_loss, x_win, y_win)

            global_idx += len(x_win)
            if self.verbose:
                print(f"  Window {i + 1}/{n_full} | Loss: {avg_window_loss:.4f} | Acc: {test_acc:.4f}")

        total_acc = total_correct / total_seen if total_seen > 0 else 0.0
        macro_f1 = f1_score(all_true, all_preds, average='macro')
        
        print(f"\nTraining complete. Overall accuracy: {total_acc:.4f}, Macro F1: {macro_f1:.4f}")
        
        return {
            "prediction_results": {'actual_labels': all_true, 'prediction_labels': all_preds}, 
            "window_accuracies": window_accuracies, 
            "overall_accuracy": total_acc,
            "macro_f1": macro_f1,
            "plateau_triggers": self.plateau_triggers
        }