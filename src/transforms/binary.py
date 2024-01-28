import torch
from src.binary.utils import init_projection_parameters


class BinaryTransform:
    def __init__(self, base_embedding_dim, binary_relative_dim, device, debinarize=False):
        self.device = device
        self.intercepts, self.slopes = init_projection_parameters(
            int(base_embedding_dim * binary_relative_dim),
            base_embedding_dim)
        self.slopes = torch.from_numpy(self.slopes.astype('float32')).to(device)
        self.intercepts = torch.from_numpy(self.intercepts.astype('float32')).to(device)
        self.r1 = 0
        self.r2 = 1
        self.debinarize = debinarize

    def __call__(self, x):
        x = (torch.mm(x.unsqueeze(0).to(self.device).float(), self.slopes) > self.intercepts).type(torch.FloatTensor)
        if self.debinarize:
            x[x == 0] = (self.r1 - self.r2 / 2) * torch.rand(x[x == 0].shape) + self.r2 / 2
            x[x == 1] = (self.r1 - self.r2 / 2) * torch.rand(x[x == 1].shape) + self.r2
        x = x.squeeze(0)

        return x
