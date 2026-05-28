import torch
import torch.nn as nn
from typing import Dict, Optional


class KeepLoRAHandler:
    """
    Implements KeepLoRA for COMPASS.

    Core idea:
    - Wp: principal subspace of pre-trained base weights (input-space directions).
           Extracted once before any training and never changed.
    - Mt: accumulated dominant feature directions from all learned tasks.
           Grows after each detected task boundary (plateau).
    - At each task boundary:
        1. Extract feature directions from current window → update Mt
        2. Compute full-weight gradient → project onto residual subspace → reinit A
        3. Freeze A for the new task; only B is trained.
    """

    def __init__(
        self,
        device: str = "cuda",
        energy_wp: float = 0.75,    # ϵw: fraction of weight energy retained in Wp
        energy_ft: float = 0.95,    # ϵf: fraction of feature energy retained in Mt
        lora_r: int = 128,
        max_subspace_rank: int = 64,  # cap on Wp and Mt columns (memory management)
        use_Wp: bool = True,           # ablation: apply pre-trained subspace constraint
        use_Mt: bool = True,           # ablation: accumulate task-specific directions Mt
        use_reinit: bool = True,       # ablation: reinit A from projected gradient SVD
    ):
        self.device = device
        self.energy_wp = energy_wp
        self.energy_ft = energy_ft
        self.lora_r = lora_r
        self.max_subspace_rank = max_subspace_rank
        self.use_Wp = use_Wp
        self.use_Mt = use_Mt
        self.use_reinit = use_reinit

        # Wp[layer_name]: (d_in, p)  — principal input-space directions of pre-trained weights
        self.principal_subspace: Dict[str, torch.Tensor] = {}
        # Mt[layer_name]: (d_in, k)  — accumulated task feature directions
        self.task_directions: Dict[str, torch.Tensor] = {}
        self.task_count: int = 0

        # kept for interface compatibility with engine checks
        self.old_u_matrices: Dict = {}

    # ------------------------------------------------------------------
    # Step 0  — called once after model setup, before any training
    # ------------------------------------------------------------------
    def extract_principal_subspace(self, model: nn.Module) -> None:
        """
        For every LoRA-wrapped linear layer, decompose the frozen base weight
        W (d_out × d_in) and store the top-p RIGHT singular vectors as Wp.

        Right singular vectors live in the d_in (input) space, matching the
        space of feature activations and lora_A.weight — so all projections
        are dimensionally consistent.
        """
        print("[COMPASS] Extracting Wp from pre-trained base weights...")
        count = 0
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "base_layer")):
                continue
            W = module.base_layer.weight.data.float()  # (d_out, d_in)
            try:
                _, S, Vh = torch.linalg.svd(W, full_matrices=False)
                # Vh: (k, d_in)  →  Vh.T columns are right singular vectors
                energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
                k_idx = (energy >= self.energy_wp).nonzero(as_tuple=True)[0]
                p = int(k_idx[0].item()) + 1 if len(k_idx) > 0 else len(S)
                p = min(p, self.max_subspace_rank)
                # Wp: (d_in, p)
                self.principal_subspace[name] = Vh[:p, :].T.to(self.device)
                count += 1
            except Exception as e:
                print(f"  [COMPASS] Wp SVD failed for {name}: {e}")
        print(f"[COMPASS] Wp extracted for {count} layers.")

    # ------------------------------------------------------------------
    # Step 1+2  — called at every detected task boundary
    # ------------------------------------------------------------------
    def consolidate_task(
        self,
        model: nn.Module,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        """Full task consolidation: update Mt, reinitialize A, freeze A."""
        print(f"\n[COMPASS] --- Task boundary #{self.task_count + 1} ---")
        if self.use_Mt:
            self._update_task_directions(model, x_tensor)
        if self.use_reinit:
            self._reinitialize_lora_A(model, x_tensor, y_tensor, criterion)
        else:
            self._reset_b_only(model)
        self.task_count += 1
        print(f"[COMPASS] Consolidation done. Tasks seen: {self.task_count}\n")

    # ------------------------------------------------------------------
    # Internal: update Mt from current-window feature activations
    # ------------------------------------------------------------------
    def _update_task_directions(self, model: nn.Module, x_tensor: torch.Tensor) -> None:
        """
        Register forward hooks to capture the input X to each LoRA layer.
        Project X onto the residual space (orthogonal to Wp and existing Mt),
        run SVD, and append dominant directions to Mt.
        """
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
                def _make_hook(n):
                    def _fn(mod, inp, out):
                        if inp and inp[0] is not None:
                            activations[n] = inp[0].detach().float().cpu()
                    return _fn
                hooks.append(module.register_forward_hook(_make_hook(name)))

        model.eval()
        with torch.no_grad():
            _ = model(x_tensor)
        for h in hooks:
            h.remove()

        updated = 0
        for name, X in activations.items():
            # X: (batch, seq, d_in) or (batch, d_in)  →  (N, d_in)
            X_flat = X.reshape(-1, X.shape[-1]).to(self.device)
            d_in = X_flat.shape[1]

            Wp = self.principal_subspace.get(name)      # (d_in, p)
            Mt_prev = self.task_directions.get(name)    # (d_in, k)

            # X̂ = X - X Wp Wp^T - X Mt Mt^T
            X_hat = X_flat
            if self.use_Wp and Wp is not None and Wp.shape[0] == d_in:
                Wp_d = Wp.to(X_flat.device)
                X_hat = X_hat - X_hat @ Wp_d @ Wp_d.T
            if Mt_prev is not None and Mt_prev.shape[0] == d_in:
                Mt_d = Mt_prev.to(X_flat.device)
                X_hat = X_hat - X_hat @ Mt_d @ Mt_d.T

            try:
                _, S_x, Vh_x = torch.linalg.svd(X_hat, full_matrices=False)
                energy = torch.cumsum(S_x ** 2, dim=0) / ((S_x ** 2).sum() + 1e-10)
                k_idx = (energy >= self.energy_ft).nonzero(as_tuple=True)[0]
                m = int(k_idx[0].item()) + 1 if len(k_idx) > 0 else min(8, len(S_x))
                m = min(m, self.max_subspace_rank)
                new_dirs = Vh_x[:m, :].T  # (d_in, m)

                if name not in self.task_directions:
                    self.task_directions[name] = new_dirs
                else:
                    combined = torch.cat(
                        [self.task_directions[name].to(new_dirs.device), new_dirs], dim=1
                    )
                    Uc, Sc, _ = torch.linalg.svd(combined, full_matrices=False)
                    energy_c = torch.cumsum(Sc ** 2, dim=0) / ((Sc ** 2).sum() + 1e-10)
                    k_c = (energy_c >= self.energy_ft).nonzero(as_tuple=True)[0]
                    k_c = int(k_c[0].item()) + 1 if len(k_c) > 0 else len(Sc)
                    k_c = min(k_c, self.max_subspace_rank)
                    self.task_directions[name] = Uc[:, :k_c]

                updated += 1
            except Exception as e:
                print(f"  [COMPASS] Mt update failed for {name}: {e}")

        print(f"  Mt updated for {updated} layers.")

    # ------------------------------------------------------------------
    # Internal: reinitialize A from projected gradient
    # ------------------------------------------------------------------
    def _reinitialize_lora_A(
        self,
        model: nn.Module,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        """
        Temporarily unfreeze base weights to compute the true full-weight gradient G.
        Project G onto the residual subspace: Ĝ = G - G·Wp·Wp^T - G·Mt·Mt^T
        Run SVD on Ĝ and set lora_A.weight = top-r rows of Vh (right singular vectors).
        Reset lora_B to zero. Then freeze A, unfreeze B.

        Dimensions:
          G             (d_out, d_in)
          Wp, Mt        (d_in, *)   →  right-multiply: G_hat = G - G @ Wp @ Wp.T
          lora_A.weight (r, d_in)   = Vh_g[:r, :]  ✓
          lora_B.weight (d_out, r)  reset to 0      ✓
        """
        for module in model.modules():
            if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                module.base_layer.weight.requires_grad_(True)

        model.train()
        model.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()

        reinit_count = 0
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "base_layer")):
                continue
            lora_a = module.lora_A["default"] if "default" in module.lora_A else None
            lora_b = module.lora_B["default"] if "default" in module.lora_B else None
            if lora_a is None or module.base_layer.weight.grad is None:
                continue

            G = module.base_layer.weight.grad.float()   # (d_out, d_in)
            d_in = G.shape[1]

            Wp = self.principal_subspace.get(name)      # (d_in, p)
            Mt = self.task_directions.get(name)         # (d_in, k)

            G_hat = G
            if self.use_Wp and Wp is not None and Wp.shape[0] == d_in:
                Wp_d = Wp.to(G.device)
                G_hat = G_hat - G_hat @ Wp_d @ Wp_d.T
            if Mt is not None and Mt.shape[0] == d_in:
                Mt_d = Mt.to(G.device)
                G_hat = G_hat - G_hat @ Mt_d @ Mt_d.T

            try:
                _, _, Vh_g = torch.linalg.svd(G_hat, full_matrices=False)
                r = min(self.lora_r, Vh_g.shape[0])
                with torch.no_grad():
                    lora_a.weight.data.copy_(Vh_g[:r, :].to(lora_a.weight.dtype))
                    if lora_b is not None:
                        lora_b.weight.data.zero_()
                lora_a.weight.requires_grad_(False)
                if lora_b is not None:
                    lora_b.weight.requires_grad_(True)
                reinit_count += 1
            except Exception as e:
                print(f"  [COMPASS] A reinit failed for {name}: {e}")

        model.zero_grad()

        for module in model.modules():
            if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                module.base_layer.weight.requires_grad_(False)
                module.base_layer.weight.grad = None

        print(f"  Re-initialized {reinit_count} LoRA A matrices.")

    # ------------------------------------------------------------------
    # Internal: reset B only (used when use_reinit=False)
    # ------------------------------------------------------------------
    def _reset_b_only(self, model) -> None:
        count = 0
        for module in model.modules():
            if hasattr(module, "lora_B"):
                for lora_b in module.lora_B.values():
                    with torch.no_grad():
                        lora_b.weight.data.zero_()
                count += 1
        print(f"  Reset B to zero for {count} layers (A kept, no gradient reinit).")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def ensure_only_B_trainable(self, model: nn.Module) -> None:
        """Freeze all lora_A, unfreeze all lora_B. Call after model setup."""
        for module in model.modules():
            if hasattr(module, "lora_A"):
                for lora_a in module.lora_A.values():
                    lora_a.weight.requires_grad_(False)
            if hasattr(module, "lora_B"):
                for lora_b in module.lora_B.values():
                    lora_b.weight.requires_grad_(True)