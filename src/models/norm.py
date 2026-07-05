from typing import Optional

import torch
import torch.nn as nn

########################### Helper Functions ######################################


def sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric eigendecomposition-based matrix square root for PSD matrices."""
    mat = (mat + mat.transpose(-1, -2)) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag_embed(eigvals.sqrt()) @ eigvecs.transpose(-1, -2)
 
 
def inv_sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric eigendecomposition-based inverse matrix square root."""
    mat = (mat + mat.transpose(-1, -2)) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)


def gaussian_monge_map_matrix(Sigma_k: torch.Tensor, Sigma_r: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """
    A_k such that T_k(x) = mu_r + A_k (x - mu_k) is the 2-Wasserstein-optimal
    affine map pushing N(mu_k, Sigma_k) forward to N(mu_r, Sigma_r).
 
    A_k = Sigma_k^{-1/2} (Sigma_k^{1/2} Sigma_r Sigma_k^{1/2})^{1/2} Sigma_k^{-1/2}
    """
    Sigma_k_sqrt = sqrtm_psd(Sigma_k, eps)
    Sigma_k_inv_sqrt = inv_sqrtm_psd(Sigma_k, eps)
    inner = sqrtm_psd(Sigma_k_sqrt @ Sigma_r @ Sigma_k_sqrt, eps)
    return Sigma_k_inv_sqrt @ inner @ Sigma_k_inv_sqrt

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


################### Fixed-point iteration: Wasserstein Barycenter ###################

def gaussian_wasserstein_barycenter(mus: torch.Tensor, sigmas: torch.Tensor,
                                     weights: torch.Tensor = None,
                                     n_iter: int = 30, eps: float = 1e-6):
    """
    mus: (K, d), sigmas: (K, d, d). Returns (mu_r, Sigma_r).
    """
    K, d = mus.shape
    device = mus.device

    # weighted barycenter
    if weights is None:
        weights = torch.ones(K, device=device) / K
 
    mu_r = (weights.unsqueeze(-1) * mus).sum(0)
    Sigma_r = sigmas.mean(0)
 
    for _ in range(n_iter):
        Sigma_r_sqrt = sqrtm_psd(Sigma_r, eps)
        Sigma_r_sqrt_inv = inv_sqrtm_psd(Sigma_r, eps)
        S = torch.zeros_like(Sigma_r)
        for k in range(K):
            S = S + weights[k] * sqrtm_psd(Sigma_r_sqrt @ sigmas[k] @ Sigma_r_sqrt, eps)
        Sigma_r = Sigma_r_sqrt_inv @ S @ S @ Sigma_r_sqrt_inv
        Sigma_r = (Sigma_r + Sigma_r.transpose(-1, -2)) / 2
 
    return mu_r, Sigma_r

class MongeNormLayer(nn.Module):
    """
    Domain-specific Gaussian Monge normalization layer.
 
    running_mu / running_sigma: EMA per-domain statistics, updated every
        training step.
    mu_r / sigma_r / A: reference barycenter and transport matrices, recomputed
        only every `recompute_every` steps

    """
    def __init__(self, feature_dim: int, num_domains: int, momentum: float = 0.05,
                 recompute_every: int = 50, eps: float = 1e-5, 
                 num_train_samples: Optional[int] = None):
        super().__init__()
        self.d = feature_dim
        self.K = num_domains
        self.momentum = momentum
        self.recompute_every = recompute_every
        self.eps = eps
 
        self.register_buffer('running_mu', torch.zeros(num_domains, feature_dim)) #(K, d)
        self.register_buffer('running_sigma',
                              torch.eye(feature_dim).unsqueeze(0).repeat(num_domains, 1, 1)) #(K, d, d)
        self.register_buffer('domain_initialized', torch.zeros(num_domains, dtype=torch.bool))
 
        self.register_buffer('mu_r', torch.zeros(feature_dim))
        self.register_buffer('sigma_r', torch.eye(feature_dim))
        self.register_buffer('A', torch.eye(feature_dim).unsqueeze(0).repeat(num_domains, 1, 1))
    
        self.num_train_samples = num_train_samples
        if num_train_samples is not None:
            self.register_buffer('sorted_train_costs', torch.zeros(num_train_samples))
            self.register_buffer('cost_dist_ready', torch.tensor(False))
        self._step_count = 0
    
    @property
    def ready(self) -> bool:
        """True once every domain has initialized"""
        return bool(self.domain_initialized.all())
    
    @torch.no_grad()
    def empirical_pit(self, d: torch.Tensor) -> torch.Tensor:
        """
        u_hat = (1 + #{i : c_(i) >= d}) / (N + 1)
        """
        if not bool(self.cost_dist_ready):
            raise RuntimeError(
                "sorted_train_costs not finalized -- call "
                "finalize_train_cost_distribution() at the end of training first."
            )
        d = torch.as_tensor(d, device=self.sorted_train_costs.device,
                            dtype=self.sorted_train_costs.dtype)
        orig_shape = d.shape
        flat = d.reshape(-1)
        idx = torch.searchsorted(self.sorted_train_costs, flat, right=False)
        N = self.sorted_train_costs.numel()
        u_hat = (N - idx + 1).to(flat.dtype) / (N + 1)
        return u_hat.reshape(orig_shape)
    
    @torch.no_grad()
    def finalize_train_cost_distribution(self, feats: torch.Tensor, eps_reg: float = 1e-5) -> None:
        if not hasattr(self, 'sorted_train_costs'):
            raise RuntimeError(
                "num_train_samples was not set at __init__; reconstruct "
                "MongeNormLayer with num_train_samples=<N> first."
            )
        if feats.shape[0] != self.sorted_train_costs.numel():
            raise ValueError(
                f"got {feats.shape[0]} samples, expected "
                f"{self.sorted_train_costs.numel()} (num_train_samples)."
            )
        C = mahalanobis_cost(feats, self.running_mu, self.running_sigma, eps_reg)  # (N, K)
        d_vals = C.min(-1).values
        sorted_costs, _ = torch.sort(d_vals)
        self.sorted_train_costs.copy_(sorted_costs)
        self.cost_dist_ready.fill_(True)
        
    @torch.no_grad()
    def update_domain_stats(self, feats: torch.Tensor, domain_ids: torch.Tensor):
        """
        feats: (B, d), domain_ids: (B,) long.
        """
        for k in domain_ids.unique():
            mask = domain_ids == k
            fk = feats[mask]
            if fk.shape[0] < 2:
                continue
            mu_batch = fk.mean(0)
            centered = fk - mu_batch
            sigma_batch = (centered.T @ centered) / (fk.shape[0] - 1)
            sigma_batch = sigma_batch + self.eps * torch.eye(self.d, device=feats.device)
 
            k = int(k)
            if not self.domain_initialized[k]:
                self.running_mu[k] = mu_batch
                self.running_sigma[k] = sigma_batch
                self.domain_initialized[k] = True
            else:
                m = self.momentum
                self.running_mu[k] = (1 - m) * self.running_mu[k] + m * mu_batch
                self.running_sigma[k] = (1 - m) * self.running_sigma[k] + m * sigma_batch

    
    @torch.no_grad()
    def refresh_transport_maps(self, force: bool = True):
        """Not start until all domains flag READY"""
        self._step_count += 1
        if not force and (self._step_count % self.recompute_every != 0):
            return
        if not self.ready:
            return
 
        mu_r, sigma_r = gaussian_wasserstein_barycenter(
            self.running_mu, self.running_sigma, n_iter=30, eps=self.eps)
        self.mu_r.copy_(mu_r)
        self.sigma_r.copy_(sigma_r)
 
        for k in range(self.K):
            self.A[k].copy_(gaussian_monge_map_matrix(self.running_sigma[k], sigma_r, self.eps))
    
    def monge_map(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        """T_k(x) = mu_r + A_k (x - mu_k), hard assignment to a single domain."""
        mu_k = self.running_mu[domain_idx]
        A_k = self.A[domain_idx]
        return self.mu_r + (x - mu_k) @ A_k.T
 
    def barycentric_map(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """hat_T(x) = sum_k w_k T_k(x). x: (B, d), weights: (B, K)."""
        centered = x.unsqueeze(1) - self.running_mu.unsqueeze(0)          # (B, K, d)
        transported = torch.einsum('bkd,kde->bke', centered, self.A.transpose(-1, -2))
        transported = transported + self.mu_r.view(1, 1, -1)
        return (weights.unsqueeze(-1) * transported).sum(1)
    
    def forward(self, x: torch.Tensor, mode: str, domain_ids: torch.Tensor = None,
                ot_weights: torch.Tensor = None, lam=None) -> torch.Tensor:
        """
        mode == 'train': domain_ids (B,) required. 
        mode == 'test':  ot_weights (B, K) and lam (scalar or (B,)) required,
        """
        if mode == 'train':
            assert domain_ids is not None, "domain_ids required in train mode"
            out = torch.zeros_like(x)
            for k in domain_ids.unique():
                mask = domain_ids == k
                out[mask] = self.monge_map(x[mask], int(k))
            return out
 
        elif mode == 'test':
            assert ot_weights is not None and lam is not None
            x_hat = self.barycentric_map(x, ot_weights)
            if torch.is_tensor(lam) and lam.dim() > 0:
                lam = lam.view(-1, 1)
            return (1 - lam) * x + lam * x_hat
 
        else:
            raise ValueError(f"unknown mode: {mode}")