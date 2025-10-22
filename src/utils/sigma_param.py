import torch


class SigmaParametrization(torch.nn.Module):
    """
    Keep U, V fixed (buffers); learn only diagonal sigma for 2D weight updates.
    Produces a delta weight: U @ diag(sigma) @ V.
    """

    def __init__(self, U: torch.Tensor, V: torch.Tensor, init_sigma: torch.Tensor):
        super().__init__()
        # store fixed bases as buffers (no grad)
        self.register_buffer("U", U.clone().detach())
        self.register_buffer("V", V.clone().detach())

        # ensure 1D sigma diag vector
        sigma_vec = init_sigma.flatten().clone().detach()
        self.sigma = torch.nn.Parameter(sigma_vec)

    def forward(self):
        # Enforce non-negativity on sigma via ReLU
        # sigma_pos = torch.relu(self.sigma)
        return self.U @ torch.diag(self.sigma) @ self.V
