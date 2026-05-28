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
from Methods.COMPASS.engine import COMPASSEngine

try:
    from Methods.COMPASS.keeplora_handler import KeepLoRAHandler
except ImportError:
    print("CRITICAL: 'keeplora_handler.py' is missing in COMPASS folder.")
    sys.exit(1)

torch.cuda.empty_cache()

MODEL_CONFIGS = {
    "distilbert/distilgpt2": {
        "hidden_size": 768,
        "target_modules": ["c_attn", "c_proj"],
        "fan_in_fan_out": True,
    },
    "HuggingFaceTB/SmolLM2-135M": {
        "hidden_size": 576,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "fan_in_fan_out": False,
    },
    "arnir0/Tiny-LLM": {
        "hidden_size": 192,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "fan_in_fan_out": False,
    },
    "Qwen/Qwen2.5-0.5B": {
        "hidden_size": 896,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "fan_in_fan_out": False,
    },
    "meta-llama/Llama-3.2-1B": {
        "hidden_size": 2048,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "fan_in_fan_out": False,
    },
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
        return self.projection(last_token_rep)


def infer_shapes_from_sampler(data_sampler):
    all_X, all_y = [], []
    if hasattr(data_sampler, "test_inputs"):
        src = data_sampler.test_inputs
        if isinstance(src, dict):
            for v in src.values():
                all_X.append(np.array(v).reshape(-1))
        elif isinstance(src, list) and len(src) > 0:
            if isinstance(src[0], (list, np.ndarray)):
                for v in src:
                    all_X.append(np.array(v).reshape(-1))
            else:
                all_X.append(np.array(src))

    n_features = int(np.max(np.concatenate(all_X))) + 1 if all_X else 0

    if hasattr(data_sampler, "test_labels"):
        src = data_sampler.test_labels
        if isinstance(src, dict):
            for v in src.values():
                all_y.append(np.array(v).reshape(-1))
        elif isinstance(src, list) and len(src) > 0:
            if isinstance(src[0], (list, np.ndarray)):
                for v in src:
                    all_y.append(np.array(v).reshape(-1))
            else:
                all_y.append(np.array(src))

    n_classes = int(np.max(np.concatenate(all_y))) + 1 if all_y else 2
    return n_features, n_classes


def run(
    dataset, model_name, window_size, epochs_per_window, learning_rate, device,
    input_dtype, plots_dir, verbose, seed, lora_r, lora_alpha,
    energy_wp, energy_ft, max_subspace_rank,
    hard_buffer_size, loss_window_length, loss_variance_threshold,
    svd_cooldown_windows, data_split, use_svd, use_oracle_drift,
    use_Wp, use_Mt, use_reinit,
):
    set_seed(seed)

    data_name, data_sampler = preprocess(dataset)
    if not hasattr(data_sampler, "case_ids"):
        raise AttributeError("Processed data missing 'case_ids'.")

    case_ids = data_sampler.case_ids
    _, idx = np.unique(case_ids, return_index=True)
    unique_cases = case_ids[np.sort(idx)]
    split_count  = int(len(unique_cases) * 0.15)
    val_cases    = unique_cases[:split_count]
    eval_cases   = unique_cases[split_count:]

    if data_split == "validation":
        print(f"\n[SPLIT] VALIDATION (first 15%)")
        mask = np.isin(case_ids, val_cases)
    elif data_split == "evaluation":
        print(f"\n[SPLIT] EVALUATION (last 85%)")
        mask = np.isin(case_ids, eval_cases)
    else:
        print(f"\n[SPLIT] FULL")
        mask = np.ones(len(case_ids), dtype=bool)

    if isinstance(data_sampler.test_inputs, dict):
        k = sorted(data_sampler.test_inputs.keys())[0]
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

    model_spec     = MODEL_CONFIGS.get(model_name, {})
    target_modules = model_spec.get("target_modules", ["q_proj", "v_proj"])
    hidden_size    = model_spec.get("hidden_size", None)
    fan_in_fan_out = model_spec.get("fan_in_fan_out", False)

    print(f"\n{data_name} | Seed={seed} | Buffer={hard_buffer_size}")
    print(f"KeepLoRA: energy_wp={energy_wp} | energy_ft={energy_ft} | max_rank={max_subspace_rank}")
    print(f"Loading: {model_name}")

    config = AutoConfig.from_pretrained(model_name)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id

    base_model  = AutoModel.from_pretrained(model_name, config=config)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        fan_in_fan_out=fan_in_fan_out,
    )
    peft_backbone = get_peft_model(base_model, peft_config)

    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", 576)

    model = LLMWithProjection(peft_backbone, hidden_size, n_classes)
    model.to(device)

    keeplora_handler = KeepLoRAHandler(
        device=device,
        energy_wp=energy_wp,
        energy_ft=energy_ft,
        lora_r=lora_r,
        max_subspace_rank=max_subspace_rank,
        use_Wp=use_Wp,
        use_Mt=use_Mt,
        use_reinit=use_reinit,
    )
    keeplora_handler.extract_principal_subspace(model)

    eng = COMPASSEngine(
        model=model,
        keeplora_handler=keeplora_handler,
        device=device,
        window_size=window_size,
        epochs_per_window=epochs_per_window,
        batch_size=32,
        learning_rate=learning_rate,
        verbose=verbose,
        plot_dir=plots_dir,
        data_name=data_name,
        model_name=model_name,                          # NEW: passed through for filename
        input_dtype=input_dtype,
        hard_buffer_size=hard_buffer_size,
        loss_window_length=loss_window_length,
        loss_variance_threshold=loss_variance_threshold,
        svd_cooldown_windows=svd_cooldown_windows,
        use_svd=use_svd,
        use_oracle_drift=use_oracle_drift,
    )

    start_time = time.perf_counter()
    results    = eng.run_stream(data_sampler)

    if "prediction_results" in results:
        save_granular_accuracy(
            dataset_name=data_name,
            method_name="COMPASS",
            seed=seed,
            true_labels=results["prediction_results"]["actual_labels"],
            pred_labels=results["prediction_results"]["prediction_labels"],
        )

    # Derive human-readable method tag
    if not use_svd:
        method_tag = "NoUpdate"
    elif not use_Wp and not use_Mt:
        method_tag = "COMPASS_LoRA_oracle" if use_oracle_drift else "COMPASS_LoRA_taskfree"
    elif use_Wp and not use_Mt:
        method_tag = "COMPASS_Wp_oracle" if use_oracle_drift else "COMPASS_Wp_taskfree"
    else:
        method_tag = "COMPASS_taskaware" if use_oracle_drift else "COMPASS_taskfree"

    # Write per-window accuracy to shared granular_accuracy.csv
    window_accs = results.get("window_accuracies", [])
    if window_accs:
        midpoints     = [i * window_size + window_size // 2 for i in range(len(window_accs))]
        granular_path = os.path.join(plots_dir, "granular_accuracy.csv")
        file_exists   = os.path.isfile(granular_path)
        with open(granular_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Dataset", "Method", "Seed", "Event_Index", "Accuracy"]
            )
            if not file_exists:
                writer.writeheader()
            for midpt, acc in zip(midpoints, window_accs):
                writer.writerow({
                    "Dataset":     data_name,
                    "Method":      method_tag,
                    "Seed":        seed,
                    "Event_Index": midpt,
                    "Accuracy":    round(acc, 3),
                })

    runtime_hours = (time.perf_counter() - start_time) / 3600.0
    overall_acc   = results.get("overall_accuracy", 0.0)
    macro_f1      = results.get("macro_f1",    0.0)
    weighted_f1   = results.get("weighted_f1", 0.0)
    plateau_count = results.get("plateau_triggers", 0)

    os.makedirs(plots_dir, exist_ok=True)
    log_path    = os.path.join(plots_dir, "compass_results.csv")
    file_exists = os.path.isfile(log_path)

    row = {
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":                 data_name,
        "split_mode":              data_split,
        "seed":                    seed,
        "model":                   model_name,
        "lora_r":                  lora_r,
        "lora_alpha":              lora_alpha,
        "window_size":             window_size,
        "learning_rate":           learning_rate,
        "epochs_per_window":       epochs_per_window,
        "hard_buffer":             hard_buffer_size,
        "loss_variance_threshold": loss_variance_threshold,
        "loss_window_length":      loss_window_length,
        "svd_cooldown_windows":    svd_cooldown_windows,
        "energy_wp":               energy_wp,
        "energy_ft":               energy_ft,
        "max_subspace_rank":       max_subspace_rank,
        "use_svd":                 use_svd,
        "use_oracle_drift":        use_oracle_drift,
        "use_Wp":                  use_Wp,
        "use_Mt":                  use_Mt,
        "use_reinit":              use_reinit,
        "ablation_variant":        method_tag,
        "overall_accuracy":        f"{overall_acc:.4f}",
        "macro_f1":                f"{macro_f1:.4f}",
        "weighted_f1":             f"{weighted_f1:.4f}",
        "plateau_triggers":        plateau_count,
        "runtime_hours":           f"{runtime_hours:.4f}",
    }

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",                  type=str,   required=True)
    p.add_argument("--model-name",               type=str,   default="distilbert/distilgpt2")
    p.add_argument("--window-size",              type=int,   default=100)
    p.add_argument("--epochs-per-window",        type=int,   default=10)
    p.add_argument("--lr",                       type=float, default=2e-4)
    p.add_argument("--device",                   type=str,   default="auto")
    p.add_argument("--input-dtype",              choices=["long", "float"], default="long")
    p.add_argument("--plots-dir",                type=str,   default="runs")
    p.add_argument("--quiet",                    default=False, action="store_true")
    p.add_argument("--seed",                     type=int,   default=42)
    p.add_argument("--lora-r",                   type=int,   default=256)
    p.add_argument("--lora-alpha",               type=int,   default=512)
    p.add_argument("--energy-wp",                type=float, default=0.75)
    p.add_argument("--energy-ft",                type=float, default=0.95)
    p.add_argument("--max-subspace-rank",        type=int,   default=64)
    p.add_argument("--hard-buffer-size",         type=int,   default=0)
    p.add_argument("--loss-window-length",       type=int,   default=5)
    p.add_argument("--loss-variance-threshold",  type=float, default=0.0005)
    p.add_argument("--svd-cooldown-windows",     type=int,   default=10)
    p.add_argument("--data-split",               choices=["full", "validation", "evaluation"], default="full")
    p.add_argument("--no-backward-transfer",     action="store_true")
    p.add_argument("--oracle-drift",             action="store_true")
    p.add_argument("--no-Wp",                    action="store_true",
                   help="Ablation: disable pre-trained subspace constraint (Wp)")
    p.add_argument("--no-Mt",                    action="store_true",
                   help="Ablation: disable task-direction accumulation (Mt)")
    p.add_argument("--no-reinit",                action="store_true",
                   help="Ablation: disable gradient-based LoRA-A reinitialization")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"[DEVICE] {device}")

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
        energy_wp=args.energy_wp,
        energy_ft=args.energy_ft,
        max_subspace_rank=args.max_subspace_rank,
        hard_buffer_size=args.hard_buffer_size,
        loss_window_length=args.loss_window_length,
        loss_variance_threshold=args.loss_variance_threshold,
        svd_cooldown_windows=args.svd_cooldown_windows,
        data_split=args.data_split,
        use_svd=not args.no_backward_transfer,
        use_oracle_drift=args.oracle_drift,
        use_Wp=not args.no_Wp,
        use_Mt=not args.no_Mt,
        use_reinit=not args.no_reinit,
    )