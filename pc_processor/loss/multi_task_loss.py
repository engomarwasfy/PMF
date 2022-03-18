import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, n_losses, sigma=None):
        super(MultiTaskLoss, self).__init__()
        if sigma is not None:
            self.sigma = torch.nn.Parameter(torch.Tensor(sigma))

        else:
            self.sigma = torch.nn.Parameter(torch.ones(n_losses)/n_losses)

    def forward(self, losses):
        return sum(
            loss / (2.0 * self.sigma[i].pow(2))
            + (self.sigma[i].pow(2) + 1.0).log()
            for i, loss in enumerate(losses)
        )
