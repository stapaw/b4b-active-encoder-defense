# Implementations of other loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from torchvision import datasets


# soft cross entropy
def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(
            torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights,
                      1))
    else:
        return torch.mean(
            torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


# Wasserstein:

def torch_wasserstein_loss(tensor_a, tensor_b):
    # Compute the first Wasserstein distance between two 1D distributions.
    return (torch_cdf_loss(tensor_a, tensor_b, p=1))


def torch_energy_loss(tensor_a, tensor_b):
    # Compute the energy distance between two 1D distributions.
    return ((2 ** 0.5) * torch_cdf_loss(tensor_a, tensor_b, p=2))


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)),
                                 dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(
            torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1))
    else:
        cdf_distance = torch.pow(
            torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p),
                      dim=-1), 1 / p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss


def torch_validate_distibution(tensor_a, tensor_b):
    # Zero sized dimension is not supported by pytorch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")


class wasserstein_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_a, tensor_b):
        return torch_wasserstein_loss(tensor_a, tensor_b)


# https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-training-objectives


# Soft nearest neighbours Loss:
# https://arxiv.org/pdf/1902.01889.pdf , https://twitter.com/nickfrosst/status/1093581702453231623
# https://github.com/tensorflow/similarity/tree/master/tensorflow_similarity/losses
# https://github.com/tensorflow/similarity/pull/203/commits/c7b5304be9c7df40297aa8382d28400ba94337c8#diff-6fb616049a9a9c0d7cc4dc686ec1746520039c9845fa7fbf9d291054b222ca18
# https://github.com/vimarshc/fastai_experiments/blob/master/Colab%20Notebooks/entanglement.ipynb

def build_masks(labels,
                batch_size):
    """Build masks that allows to select only the positive or negatives
    embeddings.
    Args:
        labels: 1D int `Tensor` that contains the class ids.
        batch_size: size of the batch.
    Returns:
        Tuple of Tensors containing the positive_mask and negative_mask
    """
    if np.ndim(labels) == 1:
        labels = torch.reshape(labels, (-1, 1))

    # same class mask
    positive_mask = (labels == labels.T).to(torch.bool)
    # not the same class
    negative_mask = torch.logical_not(positive_mask)

    # we need to remove the diagonal from positive mask
    diag = torch.logical_not(
        torch.diag(torch.ones(batch_size, dtype=torch.bool)))
    positive_mask = torch.logical_and(positive_mask, diag)

    return positive_mask, negative_mask


def pairwise_euclid_distance(a, b):
    STABILITY_EPS = 0.00001
    a = a.double()
    b = b.double()

    batch_a = a.shape[0]
    batch_b = b.shape[0]

    sqr_norm_a = torch.pow(a, 2).sum(dim=1).view(1,
                                                 batch_a) + STABILITY_EPS
    sqr_norm_b = torch.pow(b, 2).sum(dim=1).view(batch_b,
                                                 1) + STABILITY_EPS

    tile_1 = sqr_norm_a.repeat([batch_a, 1])
    tile_2 = sqr_norm_b.repeat([1, batch_b])

    inner_prod = torch.matmul(b, a.T) + STABILITY_EPS
    dist = tile_1 + tile_2 - 2 * inner_prod
    return dist


def soft_nn_loss(args,
                 features,
                 distance,
                 temperature=10000):
    """Computes the soft nearest neighbors loss.
    Args:
        labels: Labels associated with features. (now calculated in code below)
        features: Embedded examples.
        temperature: Controls relative importance given
                        to the pair of points.
    Returns:
        loss: loss value for the current batch.
    """
    batch_size = features.size()[0]
    n = int(features.size()[0] / args.batch_size)
    labels = torch.cat(
        [torch.arange(args.batch_size) for i in range(n)], dim=0)
    # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    eps = 1e-9
    pairwise_dist = distance(features, features)
    pairwise_dist = pairwise_dist / temperature
    negexpd = torch.exp(-pairwise_dist)

    # Mask out diagonal entries
    diag = torch.diag(torch.ones(batch_size, dtype=torch.bool))
    diag_mask = torch.logical_not(diag).float().to(args.device)
    negexpd = torch.mul(negexpd, diag_mask)

    # creating mask to sample same class neighboorhood
    pos_mask, _ = build_masks(labels, batch_size)
    pos_mask = pos_mask.type(torch.FloatTensor)
    pos_mask = pos_mask.to(args.device)

    # all class neighborhood
    alcn = torch.sum(negexpd, dim=1)

    # same class neighborhood
    sacn = torch.sum(torch.mul(negexpd, pos_mask), dim=1)

    # exclude examples with unique class from loss calculation
    excl = torch.not_equal(torch.sum(pos_mask, dim=1),
                           torch.zeros(batch_size).to(args.device))
    excl = excl.type(torch.FloatTensor).to(args.device)

    loss = torch.divide(sacn, alcn)
    loss = torch.multiply(torch.log(eps + loss), excl)
    loss = -torch.mean(loss)
    return loss


# Supervised contrastive learning
# https://arxiv.org/pdf/2004.11362.pdf
# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
# https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def neg_cosine(p, z):  # negative cosine similarity
    z = z.detach()  # stop gradient
    p = F.normalize(p, dim=1)  # l2-normalize
    z = F.normalize(z, dim=1)  # l2-normalize
    return -(p * z).sum(dim=1).mean()


def regression_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=-1)


# Barlow Twins https://arxiv.org/pdf/2103.03230.pdf

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [512] + [512]  # list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss


def barlow_loss(z1, z2, device):
    # z1, z2 are projections of two augmentations
    bn = nn.BatchNorm1d(z1.shape[1], affine=False).to(device)
    # empirical cross-correlation matrix
    c = bn(z1).T @ bn(z2)
    batch_size = z1.shape[0]
    # sum the cross-correlation matrix between all gpus
    c.div_(batch_size)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + 0.0051 * off_diag
    return loss
