import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
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
        output = model(data.to(device).float())
        if len(target.shape) > 1:
            target = target.squeeze(1)
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
    config_path="../config", config_name="config_reference_model_transformation.yaml", version_base="1.2"
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

    transform_label = cfg.benchmark.embeddings.transform.name
    MODEL_OUTPUT = embeddings_dataset_train[0][0].shape[0]

    AFFINE_TRANSFORM = 'affine'
    AFFINE_SHUFFLE_PAD_TRANSFORM = 'affine_shuffle_pad'
    SHUFFLE_PAD_TRANSFORM = 'shuffle_pad'
    BINARY_TRANSFORM = 'binary'
    RUNS = 3
    results = {}
    acc_results= []
    for _ in range(RUNS):
        if transform_label == AFFINE_TRANSFORM:
            transformA = AffineTransform(MODEL_OUTPUT,
                                         cfg.benchmark.embeddings.transform.scale,
                                         cfg.benchmark.embeddings.transform.translation)
        elif transform_label == SHUFFLE_PAD_TRANSFORM:
            transformA = PadAndShuffleTransform(MODEL_OUTPUT, int(cfg.benchmark.embeddings.transform.pad_dim))
        elif transform_label == AFFINE_SHUFFLE_PAD_TRANSFORM:
            transformA = AffineAndPadAndShuffleTransform(MODEL_OUTPUT,
                                                         int(cfg.benchmark.embeddings.transform.pad_dim),
                                                         cfg.benchmark.embeddings.transform.scale,
                                                         cfg.benchmark.embeddings.transform.translation)
        elif transform_label == BINARY_TRANSFORM:
            transformA = BinaryTransform(MODEL_OUTPUT, float(cfg.benchmark.embeddings.transform.binary_relative_dim),
                                         device=device, debinarize=cfg.benchmark.embeddings.transform.debinarize)
        else:
            raise NotImplementedError(f"{transform_label} not implemented")

        print("Transformations created...")

        # User A transformations
        transformedA_embeddings_train = get_transformed_embeddings(embeddings_dataset_train.encodings, transformA)
        transformedA_embeddings_test = get_transformed_embeddings(embeddings_dataset_test.encodings, transformA)

        print(embeddings_dataset_train.encodings[0], MODEL_OUTPUT)
        print(torch.mean(embeddings_dataset_train.encodings), torch.std(embeddings_dataset_train.encodings, dim=1), torch.sum(embeddings_dataset_train.encodings))
        transformedA_dataset_train = EncodingsToLabels(transformedA_embeddings_train, embeddings_dataset_train.labels)
        transformedA_dataset_test = EncodingsToLabels(transformedA_embeddings_test, embeddings_dataset_test.labels)

        train_loader = DataLoader(transformedA_dataset_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(transformedA_dataset_test, batch_size=64, shuffle=False)

        reference_model = Net(transformedA_dataset_train[0][0].shape[0], cfg.benchmark.embeddings.class_number).to(device)
        optimizer = optim.Adam(reference_model.parameters(), lr=cfg.benchmark.hparams.lr_reference_classifier)
        criterion = nn.CrossEntropyLoss()
        epochs = cfg.benchmark.hparams.train_epochs_reference_classifier

        # Train the neural network for multiple epochs
        best = 0
        for epoch in range(0, epochs):
            train(epoch, reference_model, train_loader, criterion, optimizer, device=device)
            test_loss, ref_acc = evaluate(reference_model, test_loader, criterion, device=device)
            if ref_acc > best:
                best=ref_acc
        acc_results.append(best)

    results["reference"] = acc_results
    results["reference_mean"] = np.mean(acc_results)
    results["reference_std"] = np.std(acc_results)
    results["runs"] = RUNS

    print(f"{cfg.benchmark} best acc: {acc_results}, mean: {np.mean(acc_results)}, std: {np.std(acc_results)}")

    if cfg.output_dir is None:
        output_dir = Path("results")
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
