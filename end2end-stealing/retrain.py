import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.data import Subset
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet import ResNetSimCLR
from models.resnet_big import SupConResNet
from simclr import SimCLR
import os
torch.manual_seed(0)
pathpre = f"/scratch/ssd004/scratch/{os.getenv('USER')}/checkpoint"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR',
                    default=f"/ssd003/home/{os.getenv('USER')}/data",
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name to train with',
                    choices=['stl10', 'cifar10', 'svhn', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128). This is the dimension of z = g(h).')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--temperaturesn', default=100, type=float,
                    help='temperature for soft nearest neighbors loss')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use')
parser.add_argument('--clear', default='False', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('--array_id', type=int, default=0, help='slurm array id.')
parser.add_argument('--changeepochs', action='store_true',
                    help='use to adjust epochs based on array id instead of queries.')

num_samples = [5000, 10000, 20000, 50000]
num_epochs = [5, 10, 25, 50]

def main():
    args = parser.parse_args()
    if args.changeepochs:
        args.epochs = num_epochs[args.array_id]
        args.samples = 50000
    else:
        args.samples = num_samples[args.array_id]
        #args.epochs = 100
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        print("Gpu count", torch.cuda.device_count())
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.losstype == "supcon":
        args.lr = 0.05
    if args.losstype == "softnn":
        args.lr = 0.001

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset, args.n_views)
    train_dataset = Subset(train_dataset, range(args.samples))
    print("len dataset", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    # Load stolen model.
    checkpoint = torch.load(
        f"{pathpre}/SimCLR/102resnet34infonceSTEAL/stolen_checkpoint_50000_infonce_svhn.pth.tar",
        map_location=args.device)  # manually set this.
    # note that the head is added (randomly initialized) since the stolen model is not trained with one.
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.anon(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    if args.losstype == "supcon":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        train_loader), eta_min=0, last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer,
                        scheduler=scheduler,
                        args=args, retrain=True, loss=args.losstype)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
