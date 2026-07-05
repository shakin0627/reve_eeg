import torch 

#################### Helper Function #######################
def inv_sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric eigendecomposition-based inverse matrix square root."""
    mat = (mat + mat.transpose(-1, -2)) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)

@torch.no_grad()
def mahalanobis_cost(x: torch.Tensor, anchors: torch.Tensor,
                      sigmas: torch.Tensor, eps_reg: float = 1e-5) -> torch.Tensor:
    """
    C_ik = (x_i - mu_k)^T Sigma_k^{-1} (x_i - mu_k)
 
    x:       (B, d)
    anchors: (K, d)     -> monge_layer.running_mu
    sigmas:  (K, d, d)  -> monge_layer.running_sigma
 
    returns: (B, K) cost matrix
    """
    diffs = x.unsqueeze(1) - anchors.unsqueeze(0)          # (B, K, d)
 
    sigma_inv_sqrt = inv_sqrtm_psd(sigmas, eps_reg)         # (K, d, d)
    sigma_inv = torch.einsum('kij,kjl->kil', sigma_inv_sqrt, sigma_inv_sqrt)
 
    tmp = torch.einsum('bkd,kde->bke', diffs, sigma_inv)    # (B, K, d)
    C = (tmp * diffs).sum(-1)                                # (B, K)
    return C

#################### Dual Uncertainty #######################
def combine_lambda(lam_A, lam_B, strategy: str = 'geometric'):
    """
    lam_A: float (domain-level scalar, broadcasts)
    lam_B: (B,) tensor (per-sample)
    """
    if strategy == 'geometric':
        return torch.sqrt(torch.clamp(lam_A * lam_B, min=0.0))
    elif strategy == 'product':
        return lam_A * lam_B
    elif strategy == 'min':
        return torch.minimum(torch.full_like(lam_B, lam_A), lam_B)
    else:
        raise ValueError(strategy)
    

######################### OT ###############################
class SinkhornOT:
    def __init__(self, epsilon: float = 0.1, n_iter: int = 50,
                 tol: float = 1e-6, verbose: bool = False):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    @torch.no_grad()
    def solve(self, x: torch.Tensor, anchors: torch.Tensor,
              sigmas: torch.Tensor = None,
              eps_scale: bool = True):
        """
        x: (B, d)
        anchors: (K, d)             -> monge_layer.running_mu
        sigmas: (K, d, d)           -> monge_layer.running_sigma
        """
        B, d = x.shape
        K = anchors.shape[0]
        device, dtype = x.device, x.dtype

        log_a = torch.full((B,), -float(torch.log(torch.tensor(B, dtype=dtype))),
                            device=device, dtype=dtype)
        
        log_b = torch.full((K,), -float(torch.log(torch.tensor(K, dtype=dtype))),
                                device=device, dtype=dtype)

        C = mahalanobis_cost(x, anchors, sigmas)  # (B, K)

        eps = self.epsilon
        if eps_scale:
            eps = max(self.epsilon * C.mean().item(), 1e-8)

        f = torch.zeros(B, device=device, dtype=dtype)
        g = torch.zeros(K, device=device, dtype=dtype)

        def log_sum_exp(M, dim):
            m = M.max(dim=dim, keepdim=True).values
            return (m + (M - m).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)

        for it in range(self.n_iter):
            lse_row = log_sum_exp((g.unsqueeze(0) - C) / eps, dim=1)
            f = eps * (log_a - lse_row)

            lse_col = log_sum_exp((f.unsqueeze(1) - C) / eps, dim=0)
            g = eps * (log_b - lse_col)

            if (it + 1) % 10 == 0 or it == self.n_iter - 1:
                log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
                row_err = (log_pi.logsumexp(dim=1) - log_a).abs().max().item()
                col_err = (log_pi.logsumexp(dim=0) - log_b).abs().max().item()
                if self.verbose:
                    print(f"[Sinkhorn|{self.ground_cost}] iter {it+1}: "
                          f"row_err={row_err:.2e}, col_err={col_err:.2e}, eps={eps:.4g}")
                if max(row_err, col_err) < self.tol:
                    break
        
        log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
        plan = log_pi.exp()

        row_weights = plan / (plan.sum(dim=1, keepdim=True) + 1e-12)
        batch_cost = (plan * C).sum() / plan.sum()

        return row_weights, batch_cost
    
######################### Gibbs softmax (B = 1, online streaming) ############
def softmax_transport_weights(x: torch.Tensor, anchors: torch.Tensor,
                               sigmas: torch.Tensor, alpha: float = 1.0,
                               eps_reg: float = 1e-5):
    """
    Unbalanced / single-marginal Gibbs weighting for the B=1 streaming path.
 
    tau = alpha * d
 
    x:       (B, d)   
    alpha:   calibrated scalar multiplier on tau = alpha * d
 
    returns:
        weights:  (B, K)  domain-attribution weights, softmax over -cost/tau
        lambda_B: (B,)    1 - normalized entropy of weights
    """
    d = anchors.shape[-1]
    tau = alpha * d
 
    C = mahalanobis_cost(x, anchors, sigmas, eps_reg)   # (B, K)
    K = C.shape[-1]
 
    weights = torch.softmax(-C / tau, dim=-1)            # (B, K)
 
    H = -(weights * (weights.clamp_min(1e-12)).log()).sum(-1)   # (B,)
    lambda_B = 1.0 - H / torch.log(torch.tensor(float(K), device=C.device, dtype=C.dtype))
 
    return weights, lambda_B
 