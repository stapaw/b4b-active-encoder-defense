import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import torch
import numpy as np
from omegaconf import omegaconf, OmegaConf
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomAffine
from tqdm import tqdm
import random
from src.config import Config
from src.data.custom_dataset import EncodingsToLabels, LimitedEncodingsDataset
from src.models.linear_head import Net, MapperNet
from src.transforms.affine import AffineTransform, AffineAndPadAndShuffleTransform, PadAndShuffleTransform
from src.transforms.binary import BinaryTransform
import json

class AddGaussianNoise:
    def __init__(self, std_value):
        self.std_value = std_value

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std_value
        x = x + noise
        return x

def get_transformed_embeddings(embeddings, transform):
    transformed_embeddings = []
    for embedding in tqdm(embeddings):
        transformed_embedding = transform(embedding)
        transformed_embeddings.append(transformed_embedding)
    return torch.stack(transformed_embeddings, dim=0)


# Train the neural network
def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


# Evaluate the neural network
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += criterion(output, target.to(device)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred).to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


@hydra.main(
    config_path="../config", config_name="config_fuzzying.yaml", version_base="1.2"
)
def run_main(cfg: Config):
    run(cfg)


def run(cfg: Config):
    if cfg.wandb.enable:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.device_id}") if torch.cuda.is_available() else torch.device("cpu")
    embeddings_dataset_test: EncodingsToLabels = torch.load(cfg.benchmark.embeddings.test_path)
    embeddings_dataset_train: EncodingsToLabels = torch.load(cfg.benchmark.embeddings.train_path)

    if cfg.normalize:
        embeddings_dataset_train.encodings = torch.nn.functional.normalize(embeddings_dataset_train.encodings, p=2, dim=1)
        embeddings_dataset_test.encodings = torch.nn.functional.normalize(embeddings_dataset_test.encodings, p=2, dim=1)

    std_values = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    results = {}
    for std_dev in std_values:
        noise_transform = AddGaussianNoise(std_dev)

        transformedA_embeddings_train = get_transformed_embeddings(embeddings_dataset_train.encodings, noise_transform)
        transformedA_embeddings_test = get_transformed_embeddings(embeddings_dataset_test.encodings, noise_transform)

        transformedA_dataset_train = EncodingsToLabels(transformedA_embeddings_train, embeddings_dataset_train.labels)
        transformedA_dataset_test = EncodingsToLabels(transformedA_embeddings_test, embeddings_dataset_test.labels)

        train_loader = DataLoader(transformedA_dataset_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(transformedA_dataset_test, batch_size=64, shuffle=False)

        reference_model = Net(embeddings_dataset_train[0][0].shape[0], cfg.benchmark.embeddings.class_number).to(
            device)
        optimizer = optim.Adam(reference_model.parameters(), lr=cfg.benchmark.hparams.lr_reference_classifier)
        criterion = nn.CrossEntropyLoss()
        epochs = cfg.benchmark.hparams.train_epochs_reference_classifier

        # Train the neural network for multiple epochs
        for epoch in range(0, epochs):
            train(epoch, reference_model, train_loader, criterion, optimizer, device=device)
            test_loss, ref_acc = evaluate(reference_model, test_loader, criterion, device=device)

        reference_model.eval()

        test_loss, acc = evaluate(reference_model, test_loader, criterion=nn.CrossEntropyLoss(), device=device)
        print(f"Eval_reference: std - {std_dev}, {acc}")
        results[std_dev] = acc

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_dist = (1 - cos(embeddings_dataset_test.encodings, transformedA_embeddings_test)).mean().item()
        print(f"Eval_mean_cos_sim: std - {std_dev}, {cos_dist}")
        results[f"avg_cos_dist_{std_dev}"] = cos_dist

    if cfg.output_dir is None:
        output_dir = Path("results/noise/")
    else:
        output_dir = Path(cfg.output_dir)
    output_dir = (
            output_dir / time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    with output_dir.joinpath("config.yml").open("w+") as f:
        OmegaConf.save(cfg, f)

    with output_dir.joinpath("results.json").open("w+") as f:
        json.dump(results, f)


if __name__ == "__main__":
    run_main()
