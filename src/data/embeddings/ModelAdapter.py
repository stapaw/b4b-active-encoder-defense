import sys
from abc import ABC, abstractmethod

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import utils
import vision_transformer as vits
from torch import nn
from torchvision import models as torchvision_models
from tqdm import tqdm

from src.data.embeddings.embeddings.DatasetAdapter import DatasetAdapter


class ModelAdapter(ABC):
    def __init__(self, dataset_adapter: DatasetAdapter):
        self.dataset_adapter = dataset_adapter

    @abstractmethod
    def build_network(self, args):
        pass

    @abstractmethod
    def prepare_dataset(self, data_path):
        pass

    @abstractmethod
    def process(self, args, val_loader, train_loader):
        pass

    @abstractmethod
    def save(
        self,
        encodings_train,
        encodings,
        dataset_train,
        dataset_val,
        encodings_labels_train,
        encodings_labels_val,
    ):
        pass


class DinoModelAdapter(ModelAdapter):
    def __init__(self, dataset_adapter: DatasetAdapter):
        super().__init__(dataset_adapter)
        self.__model = None

    def build_network(self, args):
        utils.init_distributed_mode(args)
        cudnn.benchmark = True

        # ============ building network ... ============
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            embed_dim = model.embed_dim * (
                args.n_last_blocks + int(args.avgpool_patchtokens)
            )
        # if the network is a XCiT
        elif "xcit" in args.arch:
            model = torch.hub.load(
                "facebookresearch/xcit:main", args.arch, num_classes=0
            )
            embed_dim = model.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch]()
            embed_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)
        model.cuda()
        model.eval()
        # load weights to evaluate
        utils.load_pretrained_weights(
            model,
            args.pretrained_weights,
            args.checkpoint_key,
            args.arch,
            args.patch_size,
        )
        print(f"Model {args.arch} built.")

        self.__model = model

    def prepare_dataset(self, data_path):
        return self.dataset_adapter.prepare_dataset_for_dino(data_path)

    def process(self, args, val_loader, train_loader):
        n = args.n_last_blocks
        encodings = []
        encodings_labels = []
        for images, label in tqdm(val_loader):
            images = images.to("cuda")
            with torch.no_grad():
                if "vit" in args.arch:
                    intermediate_output = self.__model.get_intermediate_layers(
                        images, n
                    )
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        output = torch.cat(
                            (
                                output.unsqueeze(-1),
                                torch.mean(
                                    intermediate_output[-1][:, 1:], dim=1
                                ).unsqueeze(-1),
                            ),
                            dim=-1,
                        )
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = self.__model(images)

            encodings.append(output.detach().to("cpu"))
            encodings_labels.append(label.to("cpu"))
        encodings = torch.cat(encodings)
        encodings_labels_val = torch.cat(encodings_labels)
        print(encodings.shape, type(encodings[0]))

        n = args.n_last_blocks
        encodings_train = []
        encodings_labels = []
        for images, label in tqdm(train_loader):
            images = images.to("cuda")
            with torch.no_grad():
                if "vit" in args.arch:
                    intermediate_output = self.__model.get_intermediate_layers(
                        images, n
                    )
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        output = torch.cat(
                            (
                                output.unsqueeze(-1),
                                torch.mean(
                                    intermediate_output[-1][:, 1:], dim=1
                                ).unsqueeze(-1),
                            ),
                            dim=-1,
                        )
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = self.__model(images)

            encodings_train.append(output.detach().to("cpu"))
            encodings_labels.append(label.to("cpu"))
        encodings_train = torch.cat(encodings_train)
        encodings_labels_train = torch.cat(encodings_labels)

        print(encodings_train.shape, type(encodings_train[0]))

        return encodings_train, encodings, encodings_labels_train, encodings_labels_val

    def save(
        self,
        encodings_train,
        encodings,
        dataset_train,
        dataset_val,
        encodings_labels_train,
        encodings_labels_val,
    ):
        self.dataset_adapter.save(
            "dino",
            encodings_train,
            encodings,
            dataset_train,
            dataset_val,
            encodings_labels_train,
            encodings_labels_val,
        )


class SimsiamModelAdapter(ModelAdapter):
    def __init__(self, dataset_adapter: DatasetAdapter):
        super().__init__(dataset_adapter)
        self.__victim_model = None

    def build_network(self, args):
        print("=> loading model '{}'".format(args.arch))
        victim_model = models.__dict__[args.arch]()
        checkpoint = torch.load("checkpoint_0099.pth.tar", map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                # remove prefix
                state_dict[k[len("module.encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        print("state dict", state_dict.keys())
        victim_model.load_state_dict(state_dict, strict=False)
        victim_model.fc = torch.nn.Identity()

        victim_model.cuda()
        victim_model.eval()
        # load weights to evaluate
        # utils.load_pretrained_weights(victim_model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        print(f"Model {args.arch} built.")

        self.__victim_model = victim_model

    def prepare_dataset(self, data_path):
        return self.dataset_adapter.prepare_dataset_for_simsiam(data_path)

    def process(self, args, val_loader, train_loader):
        encodings = []
        for images, _ in tqdm(val_loader):
            images = images.to("cuda")
            with torch.no_grad():
                output = self.__victim_model(images)

            encodings.append(output.detach().to("cpu"))
        encodings = torch.cat(encodings)
        print(encodings.shape, type(encodings[0]))

        encodings_train = []
        for images, _ in tqdm(train_loader):
            images = images.to("cuda")
            with torch.no_grad():
                output = self.__victim_model(images)

            encodings_train.append(output.detach().to("cpu"))
        encodings_train = torch.cat(encodings_train)
        print(encodings_train.shape, type(encodings_train[0]))

        return encodings_train, encodings, None, None

    def save(
        self,
        encodings_train,
        encodings,
        dataset_train,
        dataset_val,
        encodings_labels_train,
        encodings_labels_val,
    ):
        self.dataset_adapter.save(
            "simsiam",
            encodings_train,
            encodings,
            dataset_train,
            dataset_val,
            encodings_labels_train,
            encodings_labels_val,
        )
