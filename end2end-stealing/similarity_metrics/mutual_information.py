import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import pickle
from scipy import stats
from scipy.special import loggamma
import math
import argparse 


parser = argparse.ArgumentParser(description='dataset_inference')
parser.add_argument('--victim_model', type=str, help='location of the victim model')
parser.add_argument('--stolen_model', type=str, help='location of the stolen model')
parser.add_argument('--imagenet_location', type=str, help='location of the ImageNet dataset')
parser.add_argument('--num_dim', type=int, default=2048)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--sample_size', type=int, default=2000)


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Load victim model
victim_model = models.resnet50(pretrained=False, num_classes=10)

checkpoint = torch.load(args.victim_model, map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
        state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
    del state_dict[k]


log = victim_model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

victim_model.fc = torch.nn.Identity()

for name, param in victim_model.named_parameters():
    param.requires_grad = False

victim_model = victim_model.to(device)

victim_model.eval()


#Load stolen models 
stolen_model = models.resnet50(pretrained=False, num_classes=10)
checkpoint = torch.load(args.stolen_model, map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
    if k.startswith('module.backbone') and not k.startswith('module.encoder.fc'):
            # remove prefix
        state_dict[k[len("module.backbone."):]] = state_dict[k]
        # delete renamed or unused k
    del state_dict[k]


log = stolen_model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

stolen_model.fc = torch.nn.Identity()

for name, param in stolen_model.named_parameters():
    #if name not in ['fc.weight', 'fc.bias']:
    param.requires_grad = False

stolen_model = stolen_model.to(device)

stolen_model.eval()

print("Finish loading models")

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),transforms.ToTensor()])


#Load ImageNet 

def get_imagenet_data_loaders(batch_size):
  train_dataset = datasets.ImageNet(args.imagenet_location, split="train", transform=data_transforms)
 
  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=True)

  test_dataset = datasets.ImageNet(args.imagenet_location, split="val", transform=data_transforms)

  test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=True)
  return train_loader,test_loader


train_loader, test_loader = get_imagenet_data_loaders(args.batch_size)

print("Finish loading data")

def entropy(samples):
  N = len(samples)
  d = len(samples[0])
  distance = torch.cdist(samples, samples, p=2)
  distance = distance.masked_select(~torch.eye(N, dtype=bool)).view(N, N - 1)
  epsilon = 1e-10
  R = torch.min(distance, 1).values + epsilon
  Y = torch.mean(math.log(N-1) + d * torch.log(R))
  pi = 3.14
  B = (d/2) * math.log(pi) - loggamma(d/2 + 1)
  euler = 0.577
  return Y + B + euler


def mutual_information(model1, model2, sample_size):
  representation1 = torch.zeros(sample_size, args.num_dim).to(device)
  representation2 = torch.zeros(sample_size, args.num_dim).to(device)
  representation_joint = torch.zeros(sample_size, args.num_dim * 2).to(device)

  for i, (x_batch, _) in enumerate(train_loader):
    if i * args.batch_size >= sample_size:
      break
    x_batch = x_batch.to(device)
    r1 = model1(x_batch)
    r2 = model2(x_batch)
    h = len(r1)
    representation1[i * args.batch_size: i*args.batch_size+h] = r1
    representation2[i * args,batch_size: i*args.batch_size+h] = r2
    

  representation1 = representation1 - torch.mean(representation1, axis=0)
  representation2 = representation2 - torch.mean(representation2, axis=0)
  representation1 = representation1 / torch.norm(representation1)
  representation2 = representation2 / torch.norm(representation2)
  representation_joint = torch.cat((representation1, representation2), 1)
  
  representation1 = representation1.to("cpu")
  representation2 = representation2.to("cpu")
  representation_joint = representation_joint.to("cpu")


  return entropy(representation1)  + entropy(representation2) - entropy(representation_joint)


upper_bound = mutual_information(victim_model, victim_model).item()
random_model = models.resnet50(pretrained=False, num_classes=10)
lower_bound = mutual_information(victim_model, random_model).item()
def score(s):
  if s > upper_bound:
    s = upper_bound
  if s < lower_bound:
    s = lower_bound
  return (s - lower_bound) / (upper_bound - lower_bound)

mi = mutual_information(victim_model, stolen_model, args.sample_size).item()
print("victim stolen mutual information " + str(mi))
print("victim stolen mutual score " + str(score(mi)))










