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
    config_path="../config", config_name="config.yaml", version_base="1.2"
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

    if transform_label == AFFINE_TRANSFORM:
        transformA = AffineTransform(MODEL_OUTPUT,
                                     cfg.benchmark.embeddings.transform.scale,
                                     cfg.benchmark.embeddings.transform.translation)
        transformB = AffineTransform(MODEL_OUTPUT,
                                     cfg.benchmark.embeddings.transform.scale,
                                     cfg.benchmark.embeddings.transform.translation)
    elif transform_label == SHUFFLE_PAD_TRANSFORM:
        transformA = PadAndShuffleTransform(MODEL_OUTPUT, int(cfg.benchmark.embeddings.transform.pad_dim))
        transformB = PadAndShuffleTransform(MODEL_OUTPUT, int(cfg.benchmark.embeddings.transform.pad_dim))
    elif transform_label == AFFINE_SHUFFLE_PAD_TRANSFORM:
        transformA = AffineAndPadAndShuffleTransform(MODEL_OUTPUT,
                                                     int(cfg.benchmark.embeddings.transform.pad_dim),
                                                     cfg.benchmark.embeddings.transform.scale,
                                                     cfg.benchmark.embeddings.transform.translation)
        transformB = AffineAndPadAndShuffleTransform(MODEL_OUTPUT,
                                                     int(cfg.benchmark.embeddings.transform.pad_dim),
                                                     cfg.benchmark.embeddings.transform.scale,
                                                     cfg.benchmark.embeddings.transform.translation)
    elif transform_label == BINARY_TRANSFORM:
        transformA = BinaryTransform(MODEL_OUTPUT, float(cfg.benchmark.embeddings.transform.binary_relative_dim),
                                     device=device, debinarize=cfg.benchmark.embeddings.transform.debinarize)
        transformB = BinaryTransform(MODEL_OUTPUT, float(cfg.benchmark.embeddings.transform.binary_relative_dim),
                                     device=device, debinarize=cfg.benchmark.embeddings.transform.debinarize)
    else:
        raise NotImplementedError(f"{transform_label} not implemented")

    print("Transformations created...")
    results = {}
    embeddings_dataset_train.encodings = embeddings_dataset_train.encodings * 1000
    embeddings_dataset_test.encodings = embeddings_dataset_test.encodings * 1000

    # User A transformations
    transformedA_embeddings_train = get_transformed_embeddings(embeddings_dataset_train.encodings, transformA)
    transformedA_embeddings_test = get_transformed_embeddings(embeddings_dataset_test.encodings, transformA)

    print(embeddings_dataset_train.encodings[0], MODEL_OUTPUT)
    print(torch.mean(embeddings_dataset_train.encodings), torch.std(embeddings_dataset_train.encodings, dim=1), torch.sum(embeddings_dataset_train.encodings))
    transformedA_dataset_train = EncodingsToLabels(transformedA_embeddings_train, embeddings_dataset_train.labels)
    transformedA_dataset_test = EncodingsToLabels(transformedA_embeddings_test, embeddings_dataset_test.labels)

    # User B transformations
    transformedB_embeddings_train = get_transformed_embeddings(embeddings_dataset_train.encodings, transformB)
    transformedB_embeddings_test = get_transformed_embeddings(embeddings_dataset_test.encodings, transformB)

    transformedB_dataset_train = EncodingsToLabels(transformedB_embeddings_train, embeddings_dataset_train.labels)

    train_loader = DataLoader(transformedA_dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(transformedA_dataset_test, batch_size=64, shuffle=False)

    reference_model = Net(transformedA_dataset_train[0][0].shape[0], cfg.benchmark.embeddings.class_number).to(device)
    optimizer = optim.Adam(reference_model.parameters(), lr=cfg.benchmark.hparams.lr_reference_classifier)
    criterion = nn.CrossEntropyLoss()
    epochs = cfg.benchmark.hparams.train_epochs_reference_classifier

    # Train the neural network for multiple epochs
    for epoch in range(0, epochs):
        train(epoch, reference_model, train_loader, criterion, optimizer, device=device)
        test_loss, ref_acc = evaluate(reference_model, test_loader, criterion, device=device)

    results["reference"] = ref_acc

    reference_model.eval()

    # Train a mapper
    for n_items in cfg.benchmark.subsets:

        learning_rate = cfg.benchmark.hparams.lr_mapper
        batch_size = 64
        mapper_epochs = cfg.benchmark.hparams.train_epochs_mapper * int(10000/n_items)

        train_dataset = LimitedEncodingsDataset(transformedB_embeddings_train, transformedA_embeddings_train, n=n_items)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        mapper = MapperNet(transformedB_dataset_train[0][0].shape[0],
                           transformedA_dataset_train[0][0].shape[0],
                           n_hidden=MODEL_OUTPUT).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)

        print("Training mapper...")
        for epoch in range(1, mapper_epochs):
            train(epoch, mapper, train_loader, criterion, optimizer, device=device)

        reconstructed_encodings_test = []
        for encoding in tqdm(transformedB_embeddings_test):
            reconstructed_encoding = mapper(encoding.to(device))
            reconstructed_encodings_test.append(reconstructed_encoding.detach().to("cpu"))
        reconstructed_encodings_test = torch.cat(reconstructed_encodings_test).reshape(
            len(transformedB_embeddings_test), -1)

        # Evaluate mapped embeddings on fixed reference model
        test_dataset = EncodingsToLabels(reconstructed_encodings_test, embeddings_dataset_test.labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        test_loss, acc = evaluate(reference_model, test_loader, criterion=nn.CrossEntropyLoss(), device=device)
        print(f"Eval_reference: n_items - {n_items}, {acc}")
        results[n_items] = acc

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_dist = (1 - cos(reconstructed_encodings_test, transformedA_embeddings_test)).mean().item()
        print(f"Eval_mean_cos_sim: n_items - {n_items}, {cos_dist}")
        results[f"avg_cos_dist_{n_items}"] = cos_dist

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
