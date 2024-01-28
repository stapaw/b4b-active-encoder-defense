import math
import numpy as np
import random

import torch


def get_random_points(n, dim):
    space = np.linspace(0, 1, 1000)
    points = np.array([random.choices(space, k=dim) for i in range(0, n)])
    return points


def get_hyper_planes_n_vectors(n, dim):
    alpha = 2 * math.pi * np.random.rand(dim-1, n)
    slopes = np.tan(alpha)
    slopes = np.vstack([-slopes, [1]*n]) # from z = ax + by + c formula towards c = -ax - by + 1z (single slope is [-a,-b,1].T vector)
    return slopes


def get_hyper_planes_intercepts(slopes, points):
    assert points.T.shape == slopes.shape
    return np.sum(slopes * points.T, axis=0).reshape(1, slopes.shape[1])


def init_projection_parameters(n, output_dim):
    slopes = get_hyper_planes_n_vectors(n, output_dim)
    points = get_random_points(n, output_dim)
    intercepts = get_hyper_planes_intercepts(slopes, points)
    return intercepts, slopes


def get_binarized_embeddings(img_embeds, intercepts, slopes):
    img_transformed_embeds = []
    for embeds in img_embeds:
        binarized = (embeds.cpu().numpy().dot(slopes) > intercepts).astype('float32')
        binarized = torch.from_numpy(binarized)
        img_transformed_embeds.append(binarized.cpu())

    img_transformed_embeds = torch.cat(img_transformed_embeds)

    return img_transformed_embeds