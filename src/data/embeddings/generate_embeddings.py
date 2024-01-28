import argparse

import torch
import torchvision.models as models
import utils
from torch.utils.data import DataLoader

from src.data.embeddings.DatasetAdapter import create_dataset_adapter
from src.data.embeddings.ModelAdapter import (
    DinoModelAdapter,
    ModelAdapter,
    SimsiamModelAdapter,
)


def generate_embeddings(args):
    dataset_adapter = create_dataset_adapter(args.dataset)
    model_adapter: ModelAdapter = (
        DinoModelAdapter(dataset_adapter)
        if args.model == "dino"
        else SimsiamModelAdapter(dataset_adapter)
    )

    model_adapter.build_network(args)

    dataset_val, dataset_train = model_adapter.prepare_dataset(args.data_path)

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    (
        encodings_train,
        encodings,
        encodings_labels_train,
        encodings_labels_val,
    ) = model_adapter.process(args, val_loader, train_loader)

    model_adapter.save(
        encodings_train,
        encodings,
        dataset_train,
        dataset_val,
        encodings_labels_train,
        encodings_labels_val,
    )


def add_dino_parser(subparsers):
    parser_dino = subparsers.add_parser(
        "dino",
        prog="Evaluation with linear classification on ImageNet",
        help="Generate embeddings with dino.",
    )

    datasets_names = ["CIFAR10", "FashionMNIST", "ImageNet", "MNIST", "stl10", "SVHN"]

    parser_dino.add_argument(
        "--dataset",
        required=True,
        choices=datasets_names,
        help="Name of the dataset." + " | ".join(datasets_names),
    )
    parser_dino.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser_dino.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
    )
    parser_dino.add_argument(
        "--arch", default="vit_small", type=str, help="Architecture"
    )
    parser_dino.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser_dino.add_argument(
        "--weights_dino",
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
    parser_dino.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser_dino.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser_dino.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser_dino.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser_dino.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser_dino.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser_dino.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser_dino.add_argument(
        "--output_dir", default="./output/", help="Path to save logs and checkpoints"
    )
    parser_dino.add_argument(
        "--num_labels",
        default=10,
        type=int,
        help="Number of labels for linear classifier",
    )
    parser_dino.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser_dino.add_argument(
        "--data_path",
        help="Data path. Default: './datasets/<value of --dataset param>/'",
        type=str,
    )


def add_simsiam_parser(parser_simsiam):
    parser_simsiam = subparsers.add_parser(
        "simsiam",
        prog="PyTorch ImageNet Training",
        help="Generate embeddings with simsiam.",
    )

    datasets_names = ["CIFAR10", "FashionMNIST", "stl10", "SVHN"]

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser_simsiam.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=datasets_names,
        help="dataset for downstream task: " + " | ".join(datasets_names),
    )
    parser_simsiam.add_argument(
        "--data", metavar="DIR", help="path to imagenet dataset"
    )
    parser_simsiam.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser_simsiam.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser_simsiam.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser_simsiam.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser_simsiam.add_argument(
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
    parser_simsiam.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial (base) learning rate",
        dest="lr",
    )
    parser_simsiam.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser_simsiam.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.)",
        dest="weight_decay",
    )
    parser_simsiam.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser_simsiam.add_argument(
        "--resume", action="store_true", help="resume from checkpoint (if present)"
    )
    parser_simsiam.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser_simsiam.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser_simsiam.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser_simsiam.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser_simsiam.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser_simsiam.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser_simsiam.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser_simsiam.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # additional configs:
    parser_simsiam.add_argument(
        "--weights_simsiam",
        default="",
        type=str,
        help="path to simsiam pretrained checkpoint",
    )
    parser_simsiam.add_argument("--lars", action="store_true", help="Use LARS")
    parser_simsiam.add_argument(
        "--epochstrain",
        default=200,
        type=int,
        metavar="N",
        help="number of epochs victim was trained with",
    )
    parser_simsiam.add_argument(
        "--epochssteal",
        default=100,
        type=int,
        metavar="N",
        help="number of epochs stolen model was trained with",
    )
    parser_simsiam.add_argument(
        "--num_queries",
        default=50000,
        type=int,
        metavar="N",
        help="number of queries to steal with with",
    )
    parser_simsiam.add_argument(
        "--losstype", default="mse", type=str, help="Loss function to use."
    )
    parser_simsiam.add_argument(
        "--useval",
        default="False",
        type=str,
        help="Use validation set for stealing (only with imagenet)",
    )
    parser_simsiam.add_argument(
        "--useaug", default="False", type=str, help="Use augmentations with stealing"
    )
    parser_simsiam.add_argument(
        "--datasetsteal", default="cifar10", type=str, help="dataset used for querying"
    )
    parser_simsiam.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    parser_simsiam.add_argument(
        "--temperaturesn",
        default=1000,
        type=float,
        help="temperature for soft nearest neighbors loss",
    )
    parser_simsiam.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser_simsiam.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser_simsiam.add_argument(
        "--data_path",
        help="Data path. Default: './datasets/<value of --dataset param>/'",
        type=str,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True, dest="model")

    add_dino_parser(subparsers)
    add_simsiam_parser(subparsers)

    args = parser.parse_args()

    if args.data_path == None:
        args.data_path = "./datasets/" + args.dataset + "/"

    generate_embeddings(args)
