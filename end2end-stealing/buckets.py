import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import utils_dino
import vision_transformer as vits
from torch import nn
from torchvision import models as torchvision_models
from tqdm import tqdm
from utils import print_args


def main():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    subparsers = parser.add_subparsers(required=True, dest="model_to_steal")

    parser_dino = subparsers.add_parser("dino")
    parser_simsiam = subparsers.add_parser("simsiam")

    add_common_arguments(parser_dino)
    add_common_arguments(parser_simsiam)
    add_dino_arguments(parser_dino)

    args = parser.parse_args()

    print_args(args=args)

    buckets_covered = compute_buckets_covered(args)
    print("buckets_covered = ", buckets_covered)


def add_dino_arguments(parser_dino):
    parser_dino.add_argument(
        "--archdino",
        choices=["vit_small"],
        default="vit_small",
        type=str,
        help="Architecture",
    )
    parser_dino.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens or the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser_dino.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser_dino.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils_dino.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.	
            We typically set this to False for ViT-Small and to True with ViT-Base.""",
    )
    parser_dino.add_argument(
        "--pretrained_weights",
        default="./output/dino_deitsmall16_pretrain_full_checkpoint.pth",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser_dino.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )


def add_common_arguments(parser):
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    scratch = os.environ.get("SCRATCH") or ""
    parser.add_argument(
        "--prefix",
        default="./",
        type=str,
    )
    parser.add_argument(
        "--pathpre",
        default=scratch + "./checkpoints",
        type=str,
    )

    parser.add_argument("--alpha", default=0.8, type=float, metavar="M", help="Alpha")
    parser.add_argument("--beta", default=80, type=float, metavar="M", help="Beta")
    parser.add_argument(
        "--lam", default=0.000001, type=float, metavar="M", help="Lambda"
    )
    parser.add_argument("--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="N",
        type=int,
        # default=4096,
        default=128,
        help="mini-batch size (default: 4096), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial (base) learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument(
        "--resume", action="store_true", help="resume from checkpoint (if present)"
    )
    parser.add_argument(
        "--dataset", default="cifar10", type=str, help="dataset for downstream task"
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # additional configs:
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="path to simsiam pretrained checkpoint",
    )
    parser.add_argument("--lars", action="store_true", help="Use LARS")
    parser.add_argument(
        "--epochstrain",
        default=200,
        type=int,
        metavar="N",
        help="number of epochs victim was trained with",
    )
    parser.add_argument(
        "--epochssteal",
        default=100,
        type=int,
        metavar="N",
        help="number of epochs stolen model was trained with",
    )
    parser.add_argument(
        "--num_queries",
        default=50000,
        type=int,
        metavar="N",
        help="number of queries to steal with with",
    )
    parser.add_argument(
        "--losstype", default="mse", type=str, help="Loss function to use."
    )
    parser.add_argument(
        "--useval",
        default="False",
        type=str,
        help="Use validation set for stealing (only with imagenet)",
    )
    parser.add_argument(
        "--useaug", default="False", type=str, help="Use augmentations with stealing"
    )
    parser.add_argument(
        "--usedefence", default="False", type=str, help="Use defence by noising"
    )
    parser.add_argument(
        "--n_sybils",
        default=1,
        choices=range(1, 11),
        type=int,
        help="Number of accounts used to adaptive attack. The number can be specified from 1 to 10 (default: 1)",
    )
    parser.add_argument(
        "--num_queries_mapping",
        default=10000,
        type=int,
        metavar="N",
        help="number of queries stolen model was trained with",
    )
    parser.add_argument(
        "--datasetsteal", default="cifar10", type=str, help="dataset used for querying"
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    parser.add_argument(
        "--temperaturesn",
        default=1000,
        type=float,
        help="temperature for soft nearest neighbors loss",
    )
    parser.add_argument(
        "--num_rvecs",
        default=12,
        type=int,
        help="Number of random columns in projection matrix used to compute buckets for embedding.",
    )


def compute_buckets_covered(args, proj_count=20):
    buckets_file_path = f"{args.prefix}/resources/coverage/buckets_covered_{args.model_to_steal}_{args.datasetsteal}_{args.batch_size}.npy"
    features_file_path = f"{args.prefix}/resources/mapping_features/victim_features_{args.model_to_steal}_{args.datasetsteal}_{args.num_queries_mapping}.npz"
    try:
        print(f"Trying to load buckets from file {buckets_file_path}")
        buckets_covered = np.load(buckets_file_path)
        assert os.path.exists(features_file_path)
        return buckets_covered
    except:
        print(
            "Failed to load covered buckets OR features for training a mapper from a file. \nStarting computing new bucket coverage and gathering features for training mappers."
        )

    set_of_buckets_list = [set() for _ in range(proj_count)]
    buckets_covered = []
    proj_list = []

    victim_model = build_victim_model(args)
    train_loader = build_train_loader(args)

    victim_features_list = []
    saved_features_count = 0
    for i, (images, _) in tqdm(enumerate(train_loader)):
        images = images.to("cuda")
        with torch.no_grad():
            if args.model_to_steal == "dino" and "vit" in args.archdino:
                intermediate_output = victim_model.get_intermediate_layers(
                    images, args.n_last_blocks
                )
                victim_features = torch.cat(
                    [x[:, 0] for x in intermediate_output], dim=-1
                )
                if args.avgpool_patchtokens:
                    victim_features = torch.cat(
                        (
                            victim_features.unsqueeze(-1),
                            torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                                -1
                            ),
                        ),
                        dim=-1,
                    )
                    victim_features = victim_features.reshape(
                        victim_features.shape[0], -1
                    )
            else:
                victim_features = victim_model(images)

        victim_features = victim_features.detach().to("cpu")
        if saved_features_count < args.num_queries_mapping:
            victim_features_list.append(victim_features.numpy())
            saved_features_count += args.batch_size

        if i == 0:
            init_dim = victim_features.shape[-1]
            proj_list = [
                np.random.randn(init_dim, args.num_rvecs) for _ in range(proj_count)
            ]

        set_of_new_buckets_list = [
            get_buckets(victim_features.numpy(), proj) for proj in proj_list
        ]

        for set_of_buckets, set_of_new_buckets in zip(
            set_of_buckets_list, set_of_new_buckets_list
        ):
            set_of_buckets.update(set_of_new_buckets)

        buckets_list = [
            100 * len(set_of_buckets) / (2**args.num_rvecs)
            for set_of_buckets in set_of_buckets_list
        ]
        buckets_covered.append(np.mean(buckets_list))

    buckets_covered = np.array(buckets_covered)

    os.makedirs(f"{args.prefix}/resources/coverage", exist_ok=True)
    os.makedirs(f"{args.prefix}/resources/mapping_features", exist_ok=True)

    np.save(buckets_file_path, buckets_covered)
    np.savez(features_file_path, np.concatenate(victim_features_list))

    return buckets_covered


def build_victim_model(args):
    if args.model_to_steal == "dino":
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.archdino in vits.__dict__.keys():
            dino_model = vits.__dict__[args.archdino](
                patch_size=args.patch_size, num_classes=0
            )
            embed_dim = dino_model.embed_dim * (
                args.n_last_blocks + int(args.avgpool_patchtokens)
            )
        else:
            print(f"Unknow architecture: {args.archdino}")
            sys.exit(1)
        dino_model.cuda()
        dino_model.eval()
        # load weights to evaluate
        utils_dino.load_pretrained_weights(
            dino_model,
            args.pretrained_weights,
            args.checkpoint_key,
            args.archdino,
            args.patch_size,
        )
        print(f"Model {args.archdino} built.")
        return dino_model
    elif args.model_to_steal == "simsiam":
        victim_model = models.__dict__[args.arch]()
        checkpoint = torch.load(args.prefix + args.pretrained, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                # remove prefix
                state_dict[k[len("module.encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        # print("state dict", state_dict.keys())
        victim_model.load_state_dict(state_dict, strict=False)
        victim_model.fc = torch.nn.Identity()
        victim_model.cuda()
        victim_model.eval()
        return victim_model


def build_train_loader(args):
    train_dataset, val_dataset, _ = load_dataset(args)

    if args.useval == "True" and args.dataset == "imagenet":
        query_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        query_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
    return query_loader


def load_dataset(args):
    train_dataset = None
    val_dataset = None
    val_loader = None
    if args.datasetsteal == "imagenet":
        traindir = os.path.join(args.data, "train")
        valdir = os.path.join(args.data, "val")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if args.prefix == "/ssd003":
            train_dataset = datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet_pytorch/",
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

            indxs = random.sample(range(len(train_dataset)), args.num_queries)
            train_dataset = torch.utils.data.Subset(train_dataset, indxs)

            val_dataset = datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet_pytorch/",
                split="val",
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

    elif args.datasetsteal == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset_raw = datasets.CIFAR10(
            root=args.prefix + "/datasets/cifar10",
            train=True,
            download=True,
            transform=transform_train,
        )

        test_dataset = datasets.CIFAR10(
            root=args.prefix + "/datasets/cifar10",
            train=False,
            download=True,
            transform=transform_train,
        )

        # use both train and test samples for queries:
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_raw, test_dataset]
        )

    elif args.datasetsteal == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR100(
            root=f"{args.prefix}/data/cifar100",
            train=True,
            download=True,
            transform=transform_train,
        )

        test_dataset = datasets.CIFAR100(
            root=f"{args.prefix}/data/cifar100",
            train=False,
            download=True,
            transform=transform_test,
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

    elif args.datasetsteal == "svhn":
        transform_svhn = transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.42997558, 0.4283771, 0.44269393),
                    (0.19630221, 0.1978732, 0.19947216),
                ),
            ]
        )

        train_dataset = datasets.SVHN(
            root=f"{args.prefix}/data/SVHN",
            split="extra",
            download=False,
            transform=transform_svhn,
        )

    elif args.datasetsteal == "stl10":
        transform_stl = transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        train_dataset = datasets.STL10(
            root=f"{args.prefix}/datasets/stl10",
            split="unlabeled",
            download=False,
            transform=transform_stl,
        )

    else:
        raise Exception(f"Unknown args.datasetsteal: {args.datasetsteal}.")

    return train_dataset, val_dataset, val_loader


def get_buckets(data, proj):
    """This function maps batch of encoder output features to a set of unique buckets via LSH"""
    result = data @ proj

    hashed = list(map(tuple, (result > 0).astype(int)))

    buckets = defaultdict(list)

    for i, row in enumerate(hashed):
        buckets[row].append(i)

    return set([str(k) for k in dict(buckets).keys()])


if __name__ == "__main__":
    main()
