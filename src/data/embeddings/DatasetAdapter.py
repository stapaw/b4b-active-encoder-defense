import os
from abc import ABC, abstractmethod

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode

from src.data.custom_dataset import EncodingsToLabels
from src.data.custom_imagenet_dataset import CustomImageNetDataset


class DatasetAdapter(ABC):
    
    @abstractmethod
    def prepare_dataset_for_dino(self, data_path):
        pass

    @abstractmethod
    def prepare_dataset_for_simsiam(self, data_path):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        pass

    @abstractmethod
    def _create_dataset(self, data_path, train_transform, val_transform):
        pass


class CIFAR10DatasetAdapter(DatasetAdapter):

    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
            #                          (0.2470, 0.2435, 0.2616)),
        ])

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
            #                          (0.2470, 0.2435, 0.2616)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        # train_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.RandomResizedCrop(224),
        #     pth_transforms.RandomHorizontalFlip(),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                          (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        # This transform train for train and test data is a bit weird, probably mistake in code of stealing, but I proceed to be consistent
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = train_transform

        # val_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.CenterCrop(224),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                      (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        return self._create_dataset(data_path, train_transform, val_transform)
    
    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
        dataset_val = datasets.CIFAR10(data_path, train=False, download=True, transform=val_transform)

        return dataset_train, dataset_val 
    
    def get_name(self):
        return "cifar10"
    
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.targets)
        embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.targets)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')



class FashionMNISTDatasetAdapter(DatasetAdapter):
    
    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Grayscale(3),
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pth_transforms.Normalize((0.1307, 0.1307, 0.1307,),
            #                          (0.3081, 0.3081, 0.3081,)),

        ])

        val_transform = pth_transforms.Compose([
            pth_transforms.Grayscale(3),
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        # train_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.RandomResizedCrop(224),
        #     pth_transforms.RandomHorizontalFlip(),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                          (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        # This transform train for train and test data is a bit weird, probably mistake in code of stealing, but I proceed to be consistent
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            pth_transforms.ToTensor(),
            pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = train_transform

        # val_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.CenterCrop(224),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                      (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        return self._create_dataset(data_path, train_transform, val_transform) 

    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transform)
        dataset_val = datasets.FashionMNIST(data_path, train=False, download=True, transform=val_transform)

        return dataset_train, dataset_val

    def get_name(self):
        return "fmnist"
    
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.targets)
        embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.targets)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')

    
class ImageNetDatasetAdapter(DatasetAdapter):
    
    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        pass

    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = CustomImageNetDataset(os.path.join(data_path, "train"), transform=train_transform)
        dataset_val = CustomImageNetDataset(os.path.join(data_path, "val"), transform=val_transform)

        return dataset_train, dataset_val
    
    def get_name(self):
        return "ImageNet"
    
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, encodings_labels_train)
        embeddings_dataset_test = EncodingsToLabels(encodings, encodings_labels_val)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')

    
class MNISTDatasetAdapter(DatasetAdapter):
    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Grayscale(3),
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.1307, 0.1307, 0.1307,),
                                    (0.3081, 0.3081, 0.3081,)),

        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Grayscale(3),
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.1307, 0.1307, 0.1307,),
                                    (0.3081, 0.3081, 0.3081,)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        pass

    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=train_transform)
        dataset_val = datasets.MNIST(data_path, train=False, download=True, transform=val_transform)

        return dataset_train, dataset_val
    
    def get_name(self):
        return "mnist"
    
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.targets)
        embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.targets)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')


class stl10DatasetAdapter(DatasetAdapter):
    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        # train_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.RandomResizedCrop(224),
        #     pth_transforms.RandomHorizontalFlip(),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                          (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        # This transform train for train and test data is a bit weird, probably mistake in code of stealing, but I proceed to be consistent
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = train_transform

        # val_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.CenterCrop(224),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                      (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = datasets.STL10(data_path, split="train", download=True, transform=train_transform)
        dataset_val = datasets.STL10(data_path, split="test", download=True, transform=val_transform)

        return dataset_train, dataset_val

    def get_name(self):
        return "stl10"

    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.labels)
        embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.labels)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')


class SVHNDatasetAdapter(DatasetAdapter):

    def prepare_dataset_for_dino(self, data_path):
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pth_transforms.Normalize((0.5, 0.5, 0.5),
            #                          (0.5, 0.5, 0.5)),
        ])

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # pth_transforms.Normalize((0.5, 0.5, 0.5),
            #                          (0.5, 0.5, 0.5)),
        ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def prepare_dataset_for_simsiam(self, data_path):
        # train_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.RandomResizedCrop(224),
        #     pth_transforms.RandomHorizontalFlip(),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                          (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        # This transform train for train and test data is a bit weird, probably mistake in code of stealing, but I proceed to be consistent
        train_transform = pth_transforms.Compose([
            pth_transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = train_transform

        # val_transform = pth_transforms.Compose([
        #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     pth_transforms.CenterCrop(224),
        #     pth_transforms.ToTensor(),
        #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
        #     #                      (0.2023, 0.1994, 0.2010))
        #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #     #                          (0.2470, 0.2435, 0.2616)),
        # ])

        return self._create_dataset(data_path, train_transform, val_transform)

    def _create_dataset(self, data_path, train_transform, val_transform):
        dataset_train = datasets.SVHN(data_path, split="train", download=True, transform=train_transform)
        dataset_val = datasets.SVHN(data_path, split="test", download=True, transform=val_transform)

        return dataset_train, dataset_val

    def get_name(self):
        return "svhn"
    
    def save(self, model_infix, encodings_train, encodings, dataset_train, dataset_val, encodings_labels_train, encodings_labels_val):
        embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.labels)
        embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.labels)

        torch.save(embeddings_dataset_test, self.get_name() + '_emb_' + model_infix + '_test_dataset.pt')
        torch.save(embeddings_dataset_train, self.get_name() + '_emb_' + model_infix + '_train_dataset.pt')


def create_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    if dataset_name == "CIFAR10":
        return CIFAR10DatasetAdapter()
    elif dataset_name == "FashionMNIST":
        return FashionMNISTDatasetAdapter()
    elif dataset_name == "ImageNet":
        return ImageNetDatasetAdapter()
    elif dataset_name == "MNIST":
        return MNISTDatasetAdapter()
    elif dataset_name == "stl10":
        return stl10DatasetAdapter()
    elif dataset_name == "SVHN":
        return SVHNDatasetAdapter()
    else:
        return None