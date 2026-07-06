import torch

class LambdaController:
    """
    Online EMA controller for belief update
    """
    from norm import MongeNormLayer
    def __init__(self, monge_layer: MongeNormLayer, momentum: float = 0.05):
        self.layer = monge_layer
        self.momentum = momentum
        self.lambda_A = torch.zeros((), device=monge_layer.running_mu.device)

    def reset(self):
        self.lambda_A = torch.zeros((), device=self.lambda_A.device)

    @torch.no_grad()
    def step(self, C: torch.Tensor) -> torch.Tensor:
        d_t = C.min(-1).values                       # (B,)
        u_hat = self.layer.empirical_pit(d_t)         # (B,), Vovk-smoothed
        e_t = (1.0 - u_hat).mean()

        m = self.momentum
        self.lambda_A = (1 - m) * self.lambda_A + m * e_t
        return self.lambda_A