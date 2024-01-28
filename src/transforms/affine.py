import torch


class AffineTransform:
    def __init__(self, n, scale=True, translation=True):
        r1 = -1
        r2 = 1
        scale = torch.FloatTensor(1, n).uniform_(r1, r2) if scale else 1
        translation = torch.FloatTensor(1, n).uniform_(r1, r2) if translation else 0
        self.scale = scale
        self.translation = translation

    def __call__(self, x):
        # Scaling
        x = x.unsqueeze(0) * self.scale

        # Translation
        x = x + self.translation
        x = x.squeeze(0)

        return x


class PadAndShuffleTransform:
    def __init__(self, base_dim, pad_dim):
        self.pad_dim = pad_dim
        self.shuffled_indices = torch.randperm(base_dim + pad_dim)
        self.additional_dims = torch.randn(self.pad_dim)

    def __call__(self, x):
        # Add n additional random dimensions to the embedding
        additional_dims = self.additional_dims
        x = torch.cat((x, additional_dims), dim=0)

        x = x[self.shuffled_indices]
        return x


class AffineAndPadAndShuffleTransform:
    def __init__(self, base_dim, pad_dim, scale=True, translation=True):
        self.pad_dim = pad_dim
        self.affine_transform = AffineTransform(base_dim, scale, translation)
        self.pad_shuffle_transform = PadAndShuffleTransform(base_dim, pad_dim)

    def __call__(self, x):
        x = self.affine_transform(x)
        x = self.pad_shuffle_transform(x)
        return x
