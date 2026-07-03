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
    def __init__(self):
        super().__init__()