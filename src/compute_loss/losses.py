import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log‑Likelihood Loss with log‑variance clamping.

    For a single target value y and predicted distribution N(μ, σ²),
    the loss is:
        0.5 * ( logvar  +  (y − μ)² / exp(logvar) )

    Clamping logvar prevents numerical instability.
    """
    def __init__(self, clamp_min: float=-10.0, clamp_max:float=5.0):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor)->torch.Tensor:
        logvar = torch.clamp(logvar, self.clamp_min, self.clamp_max)
        precision = torch.exp(-logvar)
        loss = 0.5 * (logvar + (target - mu) ** 2 * precision)
        return loss.mean()
    
class MultiTaskLoss(nn.Module):
    """
    Weighted sum of three losses
    Parameters
    lambda_price : float
    lambda_dir  : float
    lambda_vol  : float
    logvar_clamp_min : float
    logvar_clamp_max : float
    """

    def __init__(
        self,lambda_price: float = 1.0,lambda_dir: float = 0.5,lambda_vol: float = 0.5,logvar_clamp_min: float = -10.0,
        logvar_clamp_max: float = 5.0,
    ):
        super().__init__()
        self.lambda_price = lambda_price
        self.lambda_dir = lambda_dir
        self.lambda_vol = lambda_vol

        self.price_loss = GaussianNLLLoss(logvar_clamp_min, logvar_clamp_max)
        self.dir_loss = nn.CrossEntropyLoss()
        self.vol_loss = nn.MSELoss()

    def forward(
        self,price_mu: torch.Tensor,price_logvar: torch.Tensor,price_target: torch.Tensor,dir_logits: torch.Tensor,
        dir_target: torch.Tensor,vol_pred: torch.Tensor,
        vol_target: torch.Tensor,
    ):
        """
        Returns
        total_loss : scalar Tensor
        losses_dict : dict with individual loss values (detached, for logging)
        """
        loss_price = self.price_loss(price_mu, price_logvar, price_target)
        loss_dir = self.dir_loss(dir_logits, dir_target)
        loss_vol = self.vol_loss(vol_pred, vol_target)

        total = (
            self.lambda_price * loss_price
            + self.lambda_dir * loss_dir
            + self.lambda_vol * loss_vol
        )

        losses_dict = {
            "loss_price": loss_price.item(),
            "loss_dir": loss_dir.item(),
            "loss_vol": loss_vol.item(),
            "loss_total": total.item(),
        }
        return total, losses_dict