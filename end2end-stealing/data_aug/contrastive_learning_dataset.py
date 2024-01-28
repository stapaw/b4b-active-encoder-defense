from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
import os


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s,
                                              0.2 * s)
        data_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(size=size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply([color_jitter], p=0.8),
             transforms.RandomGrayscale(p=0.2),
             GaussianBlur(kernel_size=int(0.1 * size)),
             transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_simclr_pipeline_transform_resize(size, s=1):
        """Resize to larger size then return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s,
                                              0.2 * s)
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                              transforms.RandomResizedCrop(
                                                  size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(
                                                        32),
                                                    n_views),
                                                download=True),

            'cifar100': lambda: datasets.CIFAR100(self.root_folder,
                                                  train=True,
                                                  transform=ContrastiveLearningViewGenerator(
                                                      self.get_simclr_pipeline_transform_resize(
                                                          224),
                                                      n_views),
                                                  # transform=ContrastiveLearningViewGenerator(
                                                  #     self.get_simclr_pipeline_transform(
                                                  #         32),
                                                  #     n_views),
                                                  download=False),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                # transform=ContrastiveLearningViewGenerator(
                #     self.get_simclr_pipeline_transform_resize(
                #         224),
                #     n_views),   # for training larger model.
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32),  # 96
                    n_views),  # for training smaller model.
                download=False),

            'svhn': lambda: datasets.SVHN(
                self.root_folder + "/SVHN",
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform_resize(
                        224),
                    n_views),
                download=False),  # for training a larger model

            # 'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
            #                                 split='train',
            #                                 transform=ContrastiveLearningViewGenerator(
            #                                     self.get_simclr_pipeline_transform(
            #                                         32),
            #                                     n_views),
            #                                 download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        32),
                    n_views)),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception
        else:
            return dataset_fn()

    def get_test_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(
                                                        32),
                                                    # verify if we use the transform here. also need the option for multiple augmentations possibly in the main code
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
                                          split='test',
                                          transform=ContrastiveLearningViewGenerator(
                                              self.get_simclr_pipeline_transform(
                                                  32),
                                              n_views),
                                          download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='val',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        32),
                    n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception
        else:
            return dataset_fn()


class RegularDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(
                                                        32), n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
                split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
                                          split='train',
                                          transform=ContrastiveLearningViewGenerator(
                                              self.get_simclr_pipeline_transform(
                                                  32),
                                              n_views),
                                          download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='train',
                transform=ContrastiveLearningViewGenerator(
                    self.get_imagenet_transform(
                        32),
                    n_views)),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception
        else:
            return dataset_fn()

    def get_test_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(
                                                        32),
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(
                f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views),
                download=True),

            'svhn': lambda: datasets.SVHN(self.root_folder + "/SVHN",
                                          split='test',
                                          transform=ContrastiveLearningViewGenerator(
                                              self.get_simclr_pipeline_transform(
                                                  32),
                                              n_views),
                                          download=True),
            'imagenet': lambda: datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet256/",
                split='val',
                transform=ContrastiveLearningViewGenerator(
                    self.get_imagenet_transform(
                        32),
                    n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception
        else:
            return dataset_fn()


if __name__ == "__main__":
    import torch

    dataset = ContrastiveLearningDataset(
        f"/ssd003/home/{os.getenv('USER')}/data")
    query_dataset = dataset.get_dataset("cifar10", 1)
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=1, shuffle=False)
    print("image", next(iter(query_loader)))
