import os
import shutil

import torch
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'),
                  'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_victim(epochs, dataset, model, arch, loss, device, pathpre, discard_mlp=False):
    checkpoint = torch.load(
        f"{pathpre}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
        map_location=device)
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint
    new_state_dict = state_dict.copy()
    if discard_mlp: # no longer necessary as the model architecture has no backbone.fc layers
        for k in list(state_dict.keys()):
            if k.startswith('backbone.fc'):
                del new_state_dict[k]
        model.load_state_dict(new_state_dict, strict=False)
        return model
    model.load_state_dict(state_dict, strict=False)
    return model


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    print("###################################################################")
    print("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ": ", value, flush=True)
    print("ARGS FINISHED", flush=True)
    print("######################################################")

def print_args_log(args, log, get_str=False):
    # log is a logging file
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    log.info("###################################################################")
    log.info("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        log.info(keys_str)
        log.info(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            log.info(key, ": ", value)
    log.info("ARGS FINISHED")
    log.info("######################################################")



def get_stl10_data_loaders(download, shuffle=False, batch_size=64):
    train_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='train', download=download,
                                  transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test', download=download,
                                  transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=100,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=64):
    train_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=True, download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10(f"/ssd003/home/{os.getenv('USER')}/data/", train=False, download=download,
                                  transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=100,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, shuffle=False, batch_size=64):
    train_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='train', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.SVHN(f"/ssd003/home/{os.getenv('USER')}/data/SVHN", split='test', download=download,
                                  transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=100,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

