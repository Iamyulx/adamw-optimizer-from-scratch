"""
AdamW Optimizer Implementation From Scratch

This module implements the AdamW optimization algorithm used in
modern deep learning models such as Transformers.

Key concepts implemented:
- First and second moment estimation
- Bias correction
- Decoupled weight decay
"""


import torch

class AdamWFromScratch(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        super().__init__(params, defaults)
    
    
    def step(self):
        
        for group in self.param_groups:
            
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                
                m = state["m"]
                v = state["v"]
                
                state["step"] += 1
                t = state["step"]
                
                
                # update biased moments
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                
                # bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                
                # Adam update
                update = m_hat / (torch.sqrt(v_hat) + eps)
                
                
                # parameter update
                p.data.add_(update, alpha=-lr)
                
                
                # decoupled weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)
