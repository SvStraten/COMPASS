import torch
import torch.nn.functional as F
import os

class CaLoraHandler:
    def __init__(self, old_task_subspace_path=None, device="cuda", projection_threshold=0.97):
        self.device = device
        self.old_u_matrices = {} 
        self.projection_threshold = projection_threshold
        self.old_task_subspace_path = old_task_subspace_path
        
        if self.old_task_subspace_path:
            self.load_subspace()

    def load_subspace(self, path=None):
        if path is not None:
            self.old_task_subspace_path = path
        if self.old_task_subspace_path and os.path.exists(self.old_task_subspace_path):
            loaded_data = torch.load(self.old_task_subspace_path, map_location=self.device, weights_only=False)
            self.old_u_matrices = loaded_data['u_matrices']
        else:
            if path: print(f"Warning: Subspace path {path} not found.")

    @staticmethod
    def compute_paca_mask(param, enable=True):
        if not enable:
            return 1.0
            
        importance = torch.abs(param.grad * param) + 1e-10
        importance_norm = F.softmax(importance.view(-1), dim=0).view_as(importance)
        return 1.0 + importance_norm

    def apply_caga_correction(self, name, param, enable=True):
        if not enable:
            return param.grad
            
        if name not in self.old_u_matrices:
            return param.grad

        g_new = param.grad
        u_old = self.old_u_matrices[name].to(param.device)

        try:
            projection_inner = torch.matmul(g_new, u_old) 
            projection = torch.matmul(projection_inner, u_old.T) 

            norm_g = torch.norm(g_new) + 1e-8
            norm_p = torch.norm(projection)
            correlation = norm_p / norm_g

            if correlation > 0.05: 
                return g_new - projection
        except Exception:
            pass
        return g_new