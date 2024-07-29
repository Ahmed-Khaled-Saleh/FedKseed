import numpy as np
import torch
from torch.optim import Optimizer

class MeZOOptimizer(Optimizer):
    def __init__(self, 
                 params, 
                 lr= 0.000003,
                 zo_eps=0.0005,
                 candidate_seeds= None,
                 weight_decay=0.0):
        
        defaults = dict(lr=lr, 
                        zo_eps=zo_eps,
                        candidate_seeds= candidate_seeds,
                        weight_decay=weight_decay)
        
        super(MeZOOptimizer, self).__init__(params, defaults)
        
        self.candidate_seeds = candidate_seeds
        self.zo_eps = zo_eps

        

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure is required for MeZOOptimizer")
        import ipdb; ipdb.set_trace()
        print(f"Candidate seeds: {self.candidate_seeds}")
        self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0]
        
        orig_params = {}
        for group in self.param_groups:
            for p in group['params']:
                orig_params[p] = p.clone()

        # Positive perturbation
        self._perturb_parameters(scaling_factor=1)
        loss_pos = closure()

        # Restore original parameters
        self._restore_parameters(orig_params)

        # Negative perturbation
        self._perturb_parameters(scaling_factor=-1)
        loss_neg = closure()

        self.projected_grad = (loss_pos - loss_neg) / (2 * self.zo_eps)

        # Restore original parameters
        self._restore_parameters(orig_params)

        if torch.isnan(loss_pos) or torch.isnan(loss_neg):
            return loss_pos

        self._sgd_step()
        return loss_pos, self.zo_random_seed, self.projected_grad
    
    def _sgd_step(self, seed= None, grad= None):

        if seed is None:
            seed = self.zo_random_seed
        if grad is None:
            grad = self.projected_grad

        for group in self.param_groups:
            lr = group['lr']
            zo_eps = group['zo_eps']
            weight_decay = group['weight_decay']
            
            # self._add_seed_pole(local_seed_pool, zo_eps)

            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                
                torch.manual_seed(seed)
                z = torch.normal(mean=0, std=1, size=p.shape, device=p.device, dtype=p.dtype)
                
                p.grad.copy_(grad * z)

                if weight_decay != 0:
                    p.grad.add_(weight_decay, p)
                
                p.add_(p.grad, alpha=-lr)

    def _perturb_parameters(self, scaling_factor):
        for group in self.param_groups:
            zo_eps = group['zo_eps']
            for p in group['params']:
                torch.manual_seed(self.zo_random_seed)
                z = torch.normal(mean=0, std=1, size=p.shape, device=p.device, dtype=p.dtype)
                p.add_(scaling_factor * zo_eps * z)

    def _restore_parameters(self, orig_params):
        for group in self.param_groups:
            for p in group['params']:
                p.copy_(orig_params[p])

    # def _add_seed_pole(self, local_seed_pool):
    #     if local_seed_pool is not None:
    #         local_seed_pool[self.zo_random_seed] += self.projected_grad