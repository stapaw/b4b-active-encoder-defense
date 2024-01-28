import torch
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path().absolute().parent))
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from models.resnet import ResNet18, ResNet34, ResNet50, ResNetSimCLRV2
from models.resnet_simclr import ResNetSimCLR # for torchvision models, imagenet
import argparse
from tqdm import tqdm
from sklearn.preprocessing import normalize as norm_rep
from numpy import linalg as LA
from numpy import dot
import math
import random
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='L2 distance')
parser.add_argument('--dataset', default='svhn',
                    help='dataset name (for victim model)', choices=['cifar10', 'svhn', 'imagenet'])
parser.add_argument('--datasetrand', default='cifar100',
                    help='dataset used for random encoder', choices=['stl10', 'cifar10', 'svhn', 'cifar100'])
parser.add_argument('--datasetsteal', default='svhn',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn', 'imagenet'])
parser.add_argument('--datasettest', default='svhn',
                    help='dataset used for computing scores', choices=['cifar10', 'svhn', 'imagenet', 'stl10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture for victim')
parser.add_argument('--archstolen', default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture for stolen model')
parser.add_argument('--archrand', default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture for random model')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries used to steal the model.')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='Number of epochs used for training victim model.')
parser.add_argument('--epochsrand', default=100, type=int, metavar='N',
                    help='Number of epochs used for training random model.')
parser.add_argument('--epochsstolen', default=100, type=int, metavar='N',
                    help='Number of epochs used for training stolen model.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use.')
parser.add_argument('--clear', default='False', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--samplesize', default=1000, type=int,
                    help='number of samples to use')
parser.add_argument('--normalize', action='store_false',
                    help='dont normalize reps')
parser.add_argument('--center', action='store_false',
                    help='dont center reps')
parser.add_argument('--plot', action='store_true',
                    help='plot')
parser.add_argument('--p', default=['1', '2', 'inf'], type=str, action='append',
                    help='norms to use for distance')



args = parser.parse_args()
args.device = device
if args.dataset == "imagenet":
    args.arch = "resnet50"
    args.archrand = "resnet50"
    args.archstolen = "resnet50"

pathpre = f"/scratch/ssd004/scratch/{os.getenv('USER')}/checkpoint"
datapath = f"/ssd003/home/{os.getenv('USER')}/data"
print("args", args)

def lp_distance(x, y, p="2", normalize = True, center = True):
    """
    Calculate the Lp distance between two representations.

    ----------
    x, y: (n, d) ndarray
        n samples from d-dimensional representations

    p: str
        norm to use e.g. 1, 2, inf

    normalize: bool
        normalize representations

    center: bool
        center representations (mean 0, var 1)

    Returns:
    --------
    dist: float
        Lp norm between x and y

    """

    assert x.shape == y.shape
    if center:
        centerx = (x - np.mean(x, axis=1).reshape(x.shape[0], 1))/(np.std(x, axis=1).reshape(x.shape[0], 1))
        x = centerx.copy()
        centery = (y - np.mean(y, axis=1).reshape(y.shape[0], 1))/(np.std(y, axis=1).reshape(y.shape[0], 1))
        y = centery.copy()
    if normalize:
        x = norm_rep(x)
        y = norm_rep(y)
    diff = x - y
    if p == "inf":
        norm = np.inf
    else:
        norm = int(p)
    dist = LA.norm(diff, ord = norm, axis=1)
    return dist.mean(), dist.std(), dist

def cosine_similarity(x, y, normalize = True, center = True):
    """
    Calculate the cosine similarity (absolute value) between two representations.

    ----------
    x, y: (n, d) ndarray
        n samples from d-dimensional representations

    normalize: bool
        normalize representations

    center: bool
        center representations (mean 0, var 1)

    Returns:
    --------
    sim: float
        cosine similarity between x and y

    """

    assert x.shape == y.shape
    if center:
        centerx = (x - np.mean(x, axis=1).reshape(x.shape[0], 1))/(np.std(x, axis=1).reshape(x.shape[0], 1))
        x = centerx.copy()
        centery = (y - np.mean(y, axis=1).reshape(y.shape[0], 1))/(np.std(y, axis=1).reshape(y.shape[0], 1))
        y = centery.copy()
    if normalize:
        x = norm_rep(x)
        y = norm_rep(y)
    # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    sims = []
    for i in range(x.shape[0]): # similarity between corresponding indices
        cos_sim = dot(x[i], y[i]) / (LA.norm(x[i]) * LA.norm(y[i]))
        sims.append(cos_sim)
    sims = np.array(sims)
    return sims.mean(), sims.std(), sims



if args.datasettest == "cifar10":
    if args.dataset != "imagenet":
        train_dataset = datasets.CIFAR10(datapath, train=True, download=False,
                                     transform=transforms.ToTensor())
    else:
        train_dataset = datasets.CIFAR10(datapath, train=True, download=False,
                                         transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=args.workers, drop_last=False, shuffle=True)
elif args.datasettest == "svhn":
    if args.dataset != "imagenet":
        train_dataset = datasets.SVHN(datapath+"/SVHN", split='train', download=False,
                                  transform=transforms.ToTensor())
    else:
        train_dataset = datasets.SVHN(datapath + "/SVHN", split='train',
                                      download=False,
                                      transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=args.workers, drop_last=False,
                              shuffle=True)
elif args.datasettest == "imagenet":
    if args.dataset != "imagenet":
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.CenterCrop(32),
                                        transforms.ToTensor()])
        train_dataset = torchvision.datasets.ImageFolder('/scratch/ssd002/datasets/imagenet_pytorch/train', transform=transform)
    else:
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        train_dataset = torchvision.datasets.ImageFolder(
            '/scratch/ssd002/datasets/imagenet_pytorch/train',
            transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=args.workers, drop_last=False,
                              shuffle=True)

elif args.datasettest == "stl10":
    if args.dataset != "imagenet":
        dim = 32
    else:
        dim = 224
    train_dataset = datasets.STL10(f"{pathpre}/SimCLR/stl10", split='train', download=False,
                   transform=transforms.Compose(
                       [transforms.Resize(dim), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=args.workers, drop_last=False,
                              shuffle=True)

if args.dataset == "imagenet":
    victim_model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, include_mlp=False).to(args.device)
    checkpoint = torch.load(f"/ssd003/home/{os.getenv('USER')}/SimCLRSTEAL/models/checkpoint_0099-batch256.pth.tar", map_location=device)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith(
                'module.encoder.fc'):
            # remove prefix
            state_dict["backbone." + k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
else:
    victim_model = ResNetSimCLRV2(base_model=args.arch,
                                              out_dim=args.out_dim, include_mlp = False).to(args.device)
    checkpoint = torch.load(
                f"{pathpre}/SimCLR/{args.epochstrain}{args.arch}infonceTRAIN/{args.dataset}_checkpoint_{args.epochstrain}_infonce.pth.tar",
                map_location=device)
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint


victim_model.load_state_dict(state_dict, strict=False)
victim_model.eval()
print(f"Loaded {args.dataset} victim model from {pathpre}/SimCLR/{args.epochstrain}{args.arch}infonceTRAIN/{args.dataset}_checkpoint_{args.epochstrain}_infonce.pth.tar")
del state_dict
del checkpoint
# Load random and stolen models:

if args.dataset == "imagenet" and args.archstolen == "resnet50":
    stolen_model = ResNetSimCLR(base_model=args.archstolen, out_dim=args.out_dim, include_mlp=False).to(device)
else:
    stolen_model = ResNetSimCLRV2(base_model=args.archstolen,
                                  out_dim=args.out_dim, include_mlp=False).to(
        device)
if args.dataset == "imagenet" and args.archrand == "resnet50":
    random_model = ResNetSimCLR(base_model=args.archrand, out_dim=args.out_dim, include_mlp=False).to(device)
else:
    random_model = ResNetSimCLRV2(base_model=args.archrand,
                                  out_dim=args.out_dim, include_mlp=False).to(
        device)

if args.dataset == "imagenet":
    locstolen = f"{pathpre}/SimCLR/SimSiam/checkpoint_{args.datasetsteal}_{args.losstype}_{args.num_queries}.pth.tar"
    checkpoint = torch.load(
            locstolen,
            map_location=device)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('module.backbone.'):
            if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                # remove prefix
                new_state_dict[k[len("module."):]] = state_dict[k]

else:
    locstolen = f"{pathpre}/SimCLR/{args.epochsstolen}{args.archstolen}{args.losstype}STEAL/stolen_checkpoint_{args.num_queries}_{args.losstype}_{args.datasetsteal}.pth.tar"
    checkpoint = torch.load(
        locstolen,
        map_location=device)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('module.backbone.'):
            if k.startswith('module.backbone') and not k.startswith(
                    'module.backbone.fc'):
                # remove prefix
                new_state_dict[k[len("module."):]] = state_dict[k]

try:
    stolen_model.load_state_dict(state_dict, strict=True)
except:
    stolen_model.load_state_dict(new_state_dict, strict=False)
del state_dict
del checkpoint
stolen_model.eval()
print(f"Loaded stolen model from {locstolen}.")

locrandom = f"{pathpre}/SimCLR/{args.epochsrand}{args.archrand}infonceTRAIN/{args.datasetrand}_checkpoint_{args.epochsrand}_infonce.pth.tar"
checkpoint = torch.load(
        locrandom,
        map_location=device)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k in list(state_dict.keys()):
    if k.startswith('module.backbone.'):
        if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
            # remove prefix
            new_state_dict[k[len("module."):]] = state_dict[k]
random_model.load_state_dict(new_state_dict, strict=False)
random_model.eval()
del state_dict
print(f"Loaded random model from {locrandom}")

if args.arch == "resnet50":
    repsize = 2048
else:
    repsize = 512

repvic = np.zeros((args.samplesize, repsize))
repstolen = np.zeros((args.samplesize, repsize))
reprandom = np.zeros((args.samplesize, repsize))
for i, (x_batch, _) in enumerate(train_loader): # enumerate(tqdm(train_loader)):
    x_batch = x_batch.to(device)
    if i >= args.samplesize:
        break
    vic = victim_model(x_batch).cpu().detach().numpy()
    sto = stolen_model(x_batch).cpu().detach().numpy()
    ran = random_model(x_batch).cpu().detach().numpy()
    repvic[i, :] = vic.copy()
    repstolen[i, :] = sto.copy()
    reprandom[i, :] = ran.copy()

# verification
assert vic.shape[1] == repsize
assert sto.shape[1] == repsize
assert ran.shape[1] == repsize
assert i == args.samplesize

for p in args.p:
    print(f"With l_{p} norm:")

    if p == "2":
        dist_s, std_s, dist_s_all = lp_distance(repvic, repstolen, p=p,
                                                normalize=args.normalize,
                                                center=args.center)
        dist_s = dist_s/2
        std_s = std_s/2
        dist_s_all = dist_s_all/2 # normalize L2 distance to range [0,1]
        print(
            f"L2 score (V,S) (conf. interval): {1-dist_s} +- {2.576 * std_s / math.sqrt(args.samplesize)}")
    else:
        dist_s, std_s, _ = lp_distance(repvic, repstolen, p=p,
                                                normalize=args.normalize,
                                                center=args.center)
    print(f"d(V, S): {dist_s} +- {std_s}")
    print(f"d(V, S) (conf. interval): {dist_s} +- {2.576 * std_s/math.sqrt(args.samplesize)}") # 2.576 = z for 99% two sided conf. interval
    if p == "2":
        dist_r, std_r, dist_r_all = lp_distance(repvic, reprandom, p=p,
                                                normalize=args.normalize,
                                                center=args.center)
        dist_r = dist_r / 2
        std_r = std_r / 2
        dist_r_all = dist_r_all / 2
        # normalize L2 distance to range [0,1]
        print(
            f"L2 score (V,R) (conf. interval): {1 - dist_r} +- {2.576 * std_s / math.sqrt(args.samplesize)}")
    else:
        dist_r, std_r, _ = lp_distance(repvic, reprandom, p=p,
                                                normalize=args.normalize,
                                                center=args.center)
    print(f"d(V, R): {dist_r} +- {std_r}")
    print(
        f"d(V, R) (conf. interval): {dist_r} +- {2.576 * std_r / math.sqrt(args.samplesize)}")

print("With cosine similarity:")
sim_s, std_s, sim_s_all = cosine_similarity(repvic, repstolen, normalize=args.normalize, center=args.center)
print(f"sim(V, S): {sim_s} +- {std_s}")
print(f"sim(V, S) (conf. interval): {sim_s} +- {2.576 * std_s/math.sqrt(args.samplesize)}")
sim_r, std_r, sim_r_all = cosine_similarity(repvic, reprandom, normalize=args.normalize, center=args.center)
print(f"sim(V, R): {sim_r} +- {std_r}")
print(
    f"sim(V, R) (conf. interval): {sim_r} +- {2.576 * std_r / math.sqrt(args.samplesize)}")

# Obfuscate:
from obfuscation_const import perm512, perm1024, perm2048, perm4096

def choose_perm(dim):
    if dim == 512:
        perm = perm512
    elif dim == 1024:
        perm = perm1024
    elif dim == 2048:
        perm = perm2048
    elif dim == 4096:
        perm = perm4096
    else:
        raise Exception(f'Unknown dim for perm: {dim}')
    return perm


def shuffle(representations):
    dim = len(representations[0])
    perm = choose_perm(dim=dim)
    return representations[:, perm]

def pad_with_const(representations):
    dim = len(representations[0])
    num_samples = len(representations)
    pertub_matrix = torch.zeros((num_samples, 2 * dim))
    pertub_matrix = pertub_matrix.to(torch.float)
    const = 2.5
    pertub_matrix += const
    perm = choose_perm(dim=2 * dim)
    pertub_matrix = pertub_matrix.numpy()
    pertub_matrix[:, perm[0:dim]] = representations
    return pertub_matrix

def linear_transform(representations):
  scale = random.uniform(-10, 10)
  offset = random.uniform(-10, 10)
  return scale * representations + offset


print("Running obfuscation")

repstolenshuffled = shuffle(repstolen)

sim_s_shuffle, std_s_shuffle, _ = cosine_similarity(repvic, repstolenshuffled, normalize=args.normalize, center=args.center)
print(f"sim(V, S) with shuffling(conf. interval): {sim_s_shuffle} +- {2.576 * std_s_shuffle/math.sqrt(args.samplesize)}")
dist_s_shuffle, std_s_shuffle, _ = lp_distance(repvic, repstolenshuffled, p=2,
                                                normalize=args.normalize,
                                                center=args.center)
dist_s_shuffle = dist_s_shuffle/2
std_s_shuffle = std_s_shuffle/2
print(
    f"L2 score (V,S) with shuffling (conf. interval): {1-dist_s_shuffle} +- {2.576 * std_s_shuffle / math.sqrt(args.samplesize)}")


repvicextend = np.append(repvic, np.zeros((len(repvic), repvic.shape[1])), axis=1)
repstolenpad = pad_with_const(repstolen)
sim_s_pad, std_s_pad, _ = cosine_similarity(repvicextend, repstolenpad, normalize=args.normalize, center=args.center)
print(f"sim(V, S) with padding(conf. interval): {sim_s_pad} +- {2.576 * std_s_pad/math.sqrt(args.samplesize)}")
dist_s_pad, std_s_pad, _ = lp_distance(repvicextend, repstolenpad, p=2,
                                                normalize=args.normalize,
                                                center=args.center)
dist_s_pad = dist_s_pad/2
std_s_pad = std_s_pad/2
print(
    f"L2 score (V,S) with padding (conf. interval): {1-dist_s_pad} +- {2.576 * std_s_pad / math.sqrt(args.samplesize)}")

repstolentransform = linear_transform(repstolen)

sim_s_transform, std_s_transform, _ = cosine_similarity(repvic, repstolentransform, normalize=args.normalize, center=args.center)
print(f"sim(V, S) with transform(conf. interval): {sim_s_transform} +- {2.576 * std_s_transform/math.sqrt(args.samplesize)}")
dist_s_transform, std_s_transform, _ = lp_distance(repvic, repstolentransform, p=2,
                                                normalize=args.normalize,
                                                center=args.center)
dist_s_transform = dist_s_transform/2
std_s_transform = std_s_transform/2
print(
    f"L2 score (V,S) with transform (conf. interval): {1-dist_s_transform} +- {2.576 * std_s_transform / math.sqrt(args.samplesize)}")

# Plots
if args.plot:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(font_scale=1.0, style='whitegrid', rc={"grid.linewidth": 1.})
    plt.hist(dist_s_all, density=True, bins='auto', label="Stolen")
    plt.hist(dist_r_all, density=True, bins='auto', label="Independent")
    plt.title("Stolen vs Independent Model L2 Distance")
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/l2_dist_v{args.dataset}_s{args.datasetsteal}_r{args.datasetrand}.jpg")
    plt.close()
    # Convert to L2 Score: 1 - (distance above)
    dist_s_all = 1-dist_s_all
    dist_r_all = 1-dist_r_all
    plt.hist(dist_s_all, density=True, bins='auto', label="Stolen")
    plt.hist(dist_r_all, density=True, bins='auto', label="Independent")
    plt.title("Stolen vs Independent Model L2 Score")
    plt.xlabel("L2 Score")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/l2_score_v{args.dataset}_s{args.datasetsteal}_r{args.datasetrand}.jpg")
    plt.close()
    sns.set(font_scale=1.0, style='whitegrid', rc={"grid.linewidth": 1.})
    plt.hist(sim_s_all, density=True, bins='auto', label="Stolen")
    plt.hist(sim_r_all, density=True, bins='auto', label="Independent")
    plt.title("Stolen vs Independent Model Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/cos_sim_v{args.dataset}_s{args.datasetsteal}_r{args.datasetrand}.jpg")
    plt.close()
    sns.set(font_scale=1.0, style='whitegrid', rc={"grid.linewidth": 1.})
    plt.hist(np.abs(sim_s_all), density=True, bins='auto', label="Stolen")
    plt.hist(np.abs(sim_r_all), density=True, bins='auto', label="Independent")
    plt.title("Stolen vs Independent Model Cosine Similarity Score")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(f"plots/cos_score_v{args.dataset}_s{args.datasetsteal}_r{args.datasetrand}.jpg")
    plt.close()
