import torch
import math
from typing import Tuple, Dict

class PagedAdamW(torch.optim.Optimizer):
    """
    Paged AdamW optimizer for memory efficiency.
    
    Offloads optimizer states to CPU when GPU memory is limited.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        offload_to_cpu: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        self.offload_to_cpu = offload_to_cpu
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW doesn't support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Store states on CPU if offloading
                    device = 'cpu' if self.offload_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(p, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p, device=device)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Move to GPU for computation
                if self.offload_to_cpu:
                    exp_avg = exp_avg.to(p.device)
                    exp_avg_sq = exp_avg_sq.to(p.device)
                
                # Decoupled weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Move back to CPU
                if self.offload_to_cpu:
                    state['exp_avg'] = exp_avg.to('cpu')
                    state['exp_avg_sq'] = exp_avg_sq.to('cpu')
        
        return loss
