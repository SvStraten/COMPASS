import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from Utils.preprocess import preprocess
from Methods.CONAP_FM_pretrained.run import LLMWithProjection, MODEL_CONFIGS, infer_shapes_from_sampler

def get_data_loader(data_sampler, batch_size, device):
    if isinstance(data_sampler.test_inputs, dict):
        x_all = np.concatenate(list(data_sampler.test_inputs.values()))
        y_all = np.concatenate(list(data_sampler.test_labels.values()))
    elif isinstance(data_sampler.test_inputs, list):
         if len(data_sampler.test_inputs) > 0 and isinstance(data_sampler.test_inputs[0], (list, np.ndarray)):
            x_all = np.concatenate(data_sampler.test_inputs)
            y_all = np.concatenate(data_sampler.test_labels)
         else:
            x_all = np.array(data_sampler.test_inputs)
            y_all = np.array(data_sampler.test_labels)
    else:
        x_all = data_sampler.test_inputs
        y_all = data_sampler.test_labels

    x_tensor = torch.tensor(x_all, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y_all, dtype=torch.long).to(device)
    
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(1)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def extract_subspace_offline(model, dataloader, device, limit_batches, projection_threshold):
    print(f"\nExtracting Subspace from pre-trained model...")
    model.train() 
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    accumulated_grads = {}
    
    batch_count = 0
    for bx, by in dataloader:
        if batch_count >= limit_batches: break
        
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()

        for name, param in model.named_parameters():
            if "lora" in name and param.grad is not None:
                if name not in accumulated_grads:
                    accumulated_grads[name] = param.grad.detach().clone()
                else:
                    accumulated_grads[name] += param.grad.detach()
        
        model.zero_grad()
        batch_count += 1

    u_matrices = {}
    for name, grad in accumulated_grads.items():
        grad_f32 = grad.float()
        try:
            U, S, Vh = torch.linalg.svd(grad_f32, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            k_indices = (energy >= projection_threshold).nonzero(as_tuple=True)[0]
            k = k_indices[0].item() + 1 if len(k_indices) > 0 else len(S)
            u_matrices[name] = Vh[:k, :].T 
        except Exception as e:
            print(f"SVD Failed for {name}: {e}")
            
    return u_matrices

def pretrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to offline CSV file")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--model-name", type=str, default="arnir0/Tiny-LLM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--calora-energy", type=float, default=0.97)
    parser.add_argument("--calora-batches", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading offline dataset: {args.dataset}")
    data_name, data_sampler = preprocess(args.dataset)
    loader = get_data_loader(data_sampler, args.batch_size, device)
    _, n_classes = infer_shapes_from_sampler(data_sampler)

    config = AutoConfig.from_pretrained(args.model_name)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id

    base_model = AutoModel.from_pretrained(args.model_name, config=config)
    
    if args.model_name in MODEL_CONFIGS:
        target_modules = MODEL_CONFIGS[args.model_name]["target_modules"]
        hidden_size = MODEL_CONFIGS[args.model_name]["hidden_size"]
    else:
        target_modules = ["q_proj", "v_proj"]
        hidden_size = 576

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        target_modules=target_modules
    )
    
    peft_backbone = get_peft_model(base_model, peft_config)
    model = LLMWithProjection(peft_backbone, hidden_size, n_classes)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Offline Training for {args.epochs} epochs...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        batch_iter = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_iter += 1
        
        avg_loss = total_loss / batch_iter if batch_iter > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

    u_matrices = extract_subspace_offline(
        model, loader, device, 
        limit_batches=args.calora_batches, 
        projection_threshold=args.calora_energy
    )

    lora_path = os.path.join(args.output_dir, "offline_lora.pt")
    torch.save(peft_backbone.state_dict(), lora_path)
    
    subspace_path = os.path.join(args.output_dir, "offline_subspace.pt")
    torch.save({'u_matrices': u_matrices}, subspace_path)

    print(f"\nModel weights saved to: {lora_path}")
    print(f"Subspace saved to: {subspace_path}")

if __name__ == "__main__":
    pretrain()