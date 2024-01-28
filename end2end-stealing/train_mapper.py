import argparse

import os
from collections import defaultdict
from torch.utils.data import Dataset, TensorDataset
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser("Train mapper for ImageNet")
    parser.add_argument(
        "--model_to_steal",
        default="simsiam",
        choices=["dino", "simsiam"],
        type=str,
        help="Possible values: simsiam or dino",
    )
    parser.add_argument("--alpha", default=0.8, type=float, metavar="M", help="Alpha")
    parser.add_argument("--beta", default=80, type=float, metavar="M", help="Beta")
    parser.add_argument(
        "--lam", default=10 ** (-6), type=float, metavar="M", help="Lambda"
    )
    parser.add_argument(
        "--num_queries_mapping",
        default=10000,
        type=int,
        metavar="N",
        help="number of queries stolen model was trained with",
    )
    args = parser.parse_args()
    train_mappers(args)


# Define the neural network architecture
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# Generate a PyTorch dataset with the encodings before and after the transformation as input and target data, respectively
class EncodingsDataset(torch.utils.data.Dataset):
    def __init__(self, transformed_encodings, encodings):
        self.transformed_encodings = transformed_encodings
        self.encodings = encodings

    def __len__(self):
        return len(self.transformed_encodings)

    def __getitem__(self, idx):
        return self.transformed_encodings[idx], self.encodings[idx]


def train_mappers(args):
    for i in range(args.n_sybils - 1):
        train_one_mapper(i, args)


def train_one_mapper(mapper_no, args):
    print(f"Training mapper number {mapper_no}...")

    def get_n_rand(arr, n):
        return arr[np.random.choice(arr.shape[0], n, replace=False), :]

    def get_buckets(data, proj):
        result = data @ proj
        hashed = list(map(tuple, (result > 0).astype(int)))
        buckets = defaultdict(list)
        for i, row in enumerate(hashed):
            buckets[row].append(i)
        return set([str(k) for k in dict(buckets).keys()])

    def f(x):
        a = args.alpha
        s = args.lam
        b = args.beta
        return s * (np.exp(np.log(a / s) * x / b) - 1)

    def noise_dataset(embeddings, args, mean_from_n_runs=10):
        # print(embeddings.shape)
        embeddings = embeddings[: args.num_queries_mapping]
        step = 1000
        num_rvecs = 12
        init_dim = embeddings.shape[-1]
        steps = [step * j for j in range(int(embeddings.shape[0] // step))]
        # shuffle and limit to num_queries_mapping
        embeddings_rand = get_n_rand(embeddings, embeddings.shape[0])
        for i in range(mean_from_n_runs):
            proj = np.random.randn(init_dim, num_rvecs)
            # print(embeddings.shape, proj.shape)

            results = []
            for n in steps:
                arr_1 = embeddings_rand[:n]
                set1 = get_buckets(arr_1, proj)
                results.append(len(set1) / 2**num_rvecs)
            if i == 0:
                results_mean = np.array(results)
            else:
                results_mean = results_mean + np.array(results)
        results = np.array(results_mean) / mean_from_n_runs
        # print(results)
        frac_buckets = np.repeat(results, step * embeddings.shape[-1])
        # print(frac_buckets, frac_buckets.shape)
        std = f(frac_buckets * 100)
        # print(std)
        noise = np.random.normal(size=embeddings.shape) * std.reshape(embeddings.shape)
        embeddings_noised = embeddings + noise
        return embeddings_noised

    # ==============================
    N_queries = int(args.num_queries_mapping)
    features_file_path = f"{args.prefix}/resources/mapping_features/victim_features_{args.model_to_steal}_{args.datasetsteal}_{args.num_queries_mapping}.npz"

    imagenet_ds_size = 2048 if args.model_to_steal == "simsiam" else 1536
    imagenet_embeddings_factor = 1000 if args.model_to_steal == "simsiam" else 1

    imagenet_ds = np.load(features_file_path)["arr_0"].reshape(-1, imagenet_ds_size)
    noised_imagenet = noise_dataset(imagenet_ds, args, 10)

    imagenet_ds = np.load(features_file_path)["arr_0"].reshape(-1, imagenet_ds_size)
    noised_imagenet_2 = noise_dataset(imagenet_ds, args, 10)

    imagenet_embeddings = noised_imagenet[:N_queries] * imagenet_embeddings_factor
    # print(imagenet_embeddings[-1])
    # print(imagenet_embeddings.shape)

    indices = np.random.choice(len(imagenet_embeddings), size=N_queries, replace=False)
    targets_full = np.repeat(np.arange(0, 1001), 100)[:N_queries]

    encodings = imagenet_embeddings[indices]
    targets = targets_full[indices]
    encodings = torch.Tensor(encodings)

    # define random affine transform
    A = np.random.uniform(
        low=-1, high=1, size=(encodings[0].shape[0], encodings[0].shape[0])
    )
    B = np.random.uniform(low=-1, high=1, size=(encodings[0].shape[0]))

    imagenet_embeddings_2 = noised_imagenet_2[:N_queries] * imagenet_embeddings_factor

    targets_full_2 = np.repeat(np.arange(0, 1001), 100)[:N_queries]

    encodings_2 = imagenet_embeddings_2[indices]
    targets_2 = targets_full_2[indices]
    encodings_2 = torch.Tensor(encodings_2)

    transformed_encodings = []
    for encoded in tqdm(encodings_2):
        transform_encoded = np.matmul(encoded, A) + B
        transformed_encodings.append(transform_encoded)
    transformed_encodings = torch.cat(transformed_encodings)
    transformed_encodings = transformed_encodings.reshape(-1, imagenet_ds_size).type(
        torch.float32
    )
    transformed_encodings.shape, type(transformed_encodings[0])

    # ====================================

    # Set hyperparameters
    input_size = imagenet_embeddings.shape[-1]
    hidden_size = imagenet_embeddings.shape[-1]
    output_size = imagenet_embeddings.shape[-1]
    learning_rate = 0.0001
    batch_size = 64
    num_epochs = int(50 * 50000 / N_queries)
    # print(f"NUM epochs {num_epochs}")

    dataset = EncodingsDataset(transformed_encodings, encodings)
    # Load dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator1
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize the model and optimizer
    mapper = SimpleNet(input_size, hidden_size, output_size).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)

    val_min_loss = 100 if args.model_to_steal == "simsiam" else 10000
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = mapper(images.to("cuda"))
            loss = criterion(outputs, labels.to("cuda"))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, (images, labels) in enumerate(test_loader):
            # Forward pass
            val_outputs = mapper(images.to("cuda"))
            val_loss = criterion(val_outputs, labels.to("cuda"))

        if epoch % 10 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, val_Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    loss.item(),
                    val_loss.item(),
                )
            )

        if val_loss < val_min_loss:
            val_min_loss = val_loss
            os.makedirs(
                f"{args.prefix}/resources/mapper/{args.model_to_steal}", exist_ok=True
            )
            torch.save(
                mapper,
                f"{args.prefix}/resources/mapper/{args.model_to_steal}/mapper_{args.num_queries_mapping}_alpha{args.alpha}_beta{args.beta}_lambda{args.lam}_no_{mapper_no}",
            )

    # =======================
    os.makedirs(f"{args.prefix}/resources/mapper/{args.model_to_steal}", exist_ok=True)
    os.makedirs(
        f"{args.prefix}/resources/transformations/{args.model_to_steal}", exist_ok=True
    )
    np.savez(
        f"{args.prefix}/resources/transformations/{args.model_to_steal}/affine_transform_{args.num_queries_mapping}_alpha{args.alpha}_beta{args.beta}_lambda{args.lam}_no_{mapper_no}.npz",
        A=A,
        B=B,
    )


if __name__ == "__main__":
    main()
