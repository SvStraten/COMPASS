import os
import sys
import argparse
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from Utils.preprocess import preprocess
from Utils.metrics import save_granular_accuracy 
from Methods.CONAP_FM_pretrained.engine import OnlineCaLoraEngine
from Methods.CONAP_FM.calora_handler import CaLoraHandler

torch.cuda.empty_cache()

MODEL_CONFIGS = {
    "HuggingFaceTB/SmolLM2-135M": {
        "hidden_size": 576,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "Qwen/Qwen2.5-0.5B": {
        "hidden_size": 896,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "arnir0/Tiny-LLM": { 
        "hidden_size": 192, 
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "meta-llama/Llama-3.2-1B": {
        "hidden_size": 2048,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    }
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LLMWithProjection(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids):
        outputs = self.backbone(input_ids, output_hidden_states=True)
        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            last_hidden = outputs[0]
        else:
            last_hidden = outputs.hidden_states[-1]
        last_token_rep = last_hidden[:, -1, :] 
        logits = self.projection(last_token_rep)
        return logits

def infer_shapes_from_sampler(data_sampler):
    all_X = []
    if hasattr(data_sampler, 'test_inputs'):
        if isinstance(data_sampler.test_inputs, dict):
            for v in data_sampler.test_inputs.values():
                all_X.append(np.array(v).reshape(-1))
        elif isinstance(data_sampler.test_inputs, list):
             if len(data_sampler.test_inputs) > 0:
                 if isinstance(data_sampler.test_inputs[0], (list, np.ndarray)):
                     for v in data_sampler.test_inputs:
                         all_X.append(np.array(v).reshape(-1))
                 else:
                     all_X.append(np.array(data_sampler.test_inputs))

    if len(all_X) > 0:
        X_all = np.concatenate(all_X, axis=0)
        max_id = int(np.max(X_all)) 
        n_features = max_id + 1
    else:
        n_features = 0

    all_y = []
    if hasattr(data_sampler, 'test_labels'):
        if isinstance(data_sampler.test_labels, dict):
            for v in data_sampler.test_labels.values():
                all_y.append(np.array(v).reshape(-1))
        elif isinstance(data_sampler.test_labels, list):
            if len(data_sampler.test_labels) > 0:
                if isinstance(data_sampler.test_labels[0], (list, np.ndarray)):
                    for v in data_sampler.test_labels:
                         all_y.append(np.array(v).reshape(-1))
                else:
                    all_y.append(np.array(data_sampler.test_labels))
    if len(all_y) > 0:
        y_all = np.concatenate(all_y, axis=0)
        n_classes = int(np.max(y_all)) + 1
    else:
        n_classes = 2
    return n_features, n_classes

def run(
    dataset, model_name, window_size, epochs_per_window, learning_rate, device, 
    input_dtype, plots_dir, verbose, seed, lora_r, lora_alpha, calora_energy, 
    calora_batches, hard_buffer_size, loss_window_length, loss_variance_threshold, 
    svd_cooldown_windows, data_split,
    use_paca, use_caga, use_svd, use_oracle_drift,
    pretrained_lora_path=None,      
    pretrained_subspace_path=None   
):
    set_seed(seed)
    
    data_name, data_sampler = preprocess(dataset)

    if not hasattr(data_sampler, 'case_ids'):
        raise AttributeError("Processed data missing 'case_ids'.")

    case_ids = data_sampler.case_ids
    _, idx = np.unique(case_ids, return_index=True)
    unique_cases_chronological = case_ids[np.sort(idx)]
    
    n_total_cases = len(unique_cases_chronological)
    split_count = int(n_total_cases * 0.15)
    
    val_cases = unique_cases_chronological[:split_count]
    eval_cases = unique_cases_chronological[split_count:]
    
    mask = None
    if data_split == "validation":
        print(f"\n[SPLIT] Mode: VALIDATION (First 15% of Cases)")
        mask = np.isin(case_ids, val_cases)
    elif data_split == "evaluation":
        print(f"\n[SPLIT] Mode: EVALUATION (Last 85% of Cases)")
        mask = np.isin(case_ids, eval_cases)
    else:
        print(f"\n[SPLIT] Mode: FULL (All Cases)")
        mask = np.ones(len(case_ids), dtype=bool)

    if isinstance(data_sampler.test_inputs, dict):
        keys = sorted(data_sampler.test_inputs.keys())
        k = keys[0]
        data_sampler.test_inputs = {k: data_sampler.test_inputs[k][mask]}
        data_sampler.test_labels = {k: data_sampler.test_labels[k][mask]}
    elif isinstance(data_sampler.test_inputs, list):
         if len(data_sampler.test_inputs) > 0 and isinstance(data_sampler.test_inputs[0], (list, np.ndarray)):
            data_sampler.test_inputs = [data_sampler.test_inputs[0][mask]]
            data_sampler.test_labels = [data_sampler.test_labels[0][mask]]
         else:
            data_sampler.test_inputs = data_sampler.test_inputs[mask]
            data_sampler.test_labels = data_sampler.test_labels[mask]

    _, n_classes = infer_shapes_from_sampler(data_sampler)

    if model_name not in MODEL_CONFIGS:
        print(f"Warning: Model {model_name} not in config dictionary. Using default LoRA targets.")
        target_modules = ["q_proj", "v_proj"] 
        hidden_size_override = None
    else:
        model_spec = MODEL_CONFIGS[model_name]
        target_modules = model_spec["target_modules"]
        hidden_size_override = model_spec["hidden_size"]

    print(f"\n{data_name} | Seed={seed} | Data Split={data_split}")
    print(f"Components: PACA={use_paca} | CAGA={use_caga} | SVD={use_svd} | OracleDrift={use_oracle_drift}")
    print(f"Loading FM: {model_name}")

    config = AutoConfig.from_pretrained(model_name)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id

    base_model = AutoModel.from_pretrained(model_name, config=config)
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=0.05,
        target_modules=target_modules
    )
    
    peft_backbone = get_peft_model(base_model, peft_config)
    
    if pretrained_lora_path and os.path.exists(pretrained_lora_path):
        print(f"Loading pretrained LoRA weights from {pretrained_lora_path}")
        try:
            state_dict = torch.load(pretrained_lora_path, map_location=device, weights_only=True)
            missing, unexpected = peft_backbone.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                print(f"[Init] Note: {len(missing)} missing keys (Expected for projection layers).")
        except Exception as e:
            print(f"[Init] Error loading weights: {e}")
    
    if hidden_size_override:
        hidden_size = hidden_size_override
    else:
        hidden_size = getattr(config, "hidden_size", 576)

    model = LLMWithProjection(peft_backbone, hidden_size, n_classes)
    model.to(device)

    calora_handler = CaLoraHandler(
        old_task_subspace_path=pretrained_subspace_path, 
        device=device,
        projection_threshold=calora_energy
    )

    eng = OnlineCaLoraEngine(
        model=model,
        calora_handler=calora_handler,
        device=device,
        window_size=window_size,
        epochs_per_window=epochs_per_window,
        batch_size=32,
        learning_rate=learning_rate,
        verbose=verbose,
        plot_dir=plots_dir,
        data_name=data_name,
        svd_limit_batches=calora_batches,
        input_dtype=input_dtype,
        hard_buffer_size=hard_buffer_size,
        loss_window_length=loss_window_length,
        loss_variance_threshold=loss_variance_threshold,
        svd_cooldown_windows=svd_cooldown_windows,
        use_paca=use_paca,
        use_caga=use_caga,
        use_svd=use_svd,
        use_oracle_drift=use_oracle_drift
    )

    start_time = time.perf_counter()
    results = eng.run_stream(data_sampler) 
    
    if "prediction_results" in results:
        save_granular_accuracy(
            dataset_name=data_name,
            method_name="CONAP_LLM",
            seed=seed,
            true_labels=results['prediction_results']['actual_labels'],
            pred_labels=results['prediction_results']['prediction_labels']
        )

    end_time = time.perf_counter()
    
    runtime_hours = (end_time - start_time) / 3600.0
    overall_acc = results.get("overall_accuracy", 0.0)
    macro_f1 = results.get("macro_f1", 0.0)
    plateau_count = results.get("plateau_triggers", 0)

    os.makedirs(plots_dir, exist_ok=True)
    log_path = os.path.join(plots_dir, "conap_ablation_results.csv")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": data_name,
        "split_mode": data_split,
        "seed": seed,
        "use_paca": use_paca,
        "use_caga": use_caga,
        "use_svd": use_svd,
        "use_oracle_drift": use_oracle_drift,
        "pretrained": "Yes" if pretrained_lora_path else "No",
        "hard_buffer": hard_buffer_size,
        "overall_accuracy": f"{overall_acc:.4f}",
        "macro_f1": f"{macro_f1:.4f}",
        "plateau_triggers": plateau_count,
        "runtime_hours": f"{runtime_hours:.4f}",
    }

    fieldnames = list(row.keys())
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model-name", type=str, default="arnir0/Tiny-LLM")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--epochs-per-window", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--input-dtype", choices=["long", "float"], default="long")
    p.add_argument("--plots-dir", type=str, default="runs")
    p.add_argument("--quiet", default=False, action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--calora-energy", type=float, default=0.95)
    p.add_argument("--calora-batches", type=int, default=30)
    p.add_argument("--hard-buffer-size", type=int, default=0)
    p.add_argument("--loss-window-length", type=int, default=5)
    p.add_argument("--loss-variance-threshold", type=float, default=0.0005)
    p.add_argument("--svd-cooldown-windows", type=int, default=10)
    p.add_argument("--data-split", choices=["full", "validation", "evaluation"], default="full")
    p.add_argument("--pretrained-lora-path", type=str, default=None, 
                   help="Path to .pt file containing offline trained LoRA weights")
    p.add_argument("--pretrained-subspace-path", type=str, default=None, 
                   help="Path to .pt file containing offline subspace (u_matrices)")

    p.add_argument("--no-backward-transfer", action="store_true", 
                   help="Disable PACA, CAGA, and SVD simultaneously")
    p.add_argument("--oracle-drift", action="store_true", 
                   help="Use Oracle Drift points instead of Plateau Detection")
    
    p.add_argument("--no-paca", action="store_true")
    p.add_argument("--no-caga", action="store_true")
    p.add_argument("--no-svd", action="store_true")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    
    if args.no_backward_transfer:
        use_paca = False
        use_caga = False
        use_svd = False
    else:
        use_paca = not args.no_paca
        use_caga = not args.no_caga
        use_svd = not args.no_svd
        
    use_oracle_drift = args.oracle_drift
    
    run(
        dataset=args.dataset,
        model_name=args.model_name,
        window_size=args.window_size,
        epochs_per_window=args.epochs_per_window,
        learning_rate=args.lr,
        device=device,
        input_dtype=args.input_dtype,
        plots_dir=args.plots_dir,
        verbose=not args.quiet,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        calora_energy=args.calora_energy,
        calora_batches=args.calora_batches,
        hard_buffer_size=args.hard_buffer_size,
        loss_window_length=args.loss_window_length,
        loss_variance_threshold=args.loss_variance_threshold,
        svd_cooldown_windows=args.svd_cooldown_windows,
        data_split=args.data_split,
        use_paca=use_paca,
        use_caga=use_caga,
        use_svd=use_svd,
        use_oracle_drift=use_oracle_drift,
        pretrained_lora_path=args.pretrained_lora_path,
        pretrained_subspace_path=args.pretrained_subspace_path
    )