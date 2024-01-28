from torch.utils.data import Dataset, TensorDataset


class EncodingsToLabels(TensorDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]


class LimitedEncodingsDataset(Dataset):
    def __init__(self, transformed_encodings, encodings, n):
        self.transformed_encodings = transformed_encodings[:n]
        self.encodings = encodings[:n]

    def __len__(self):
        return len(self.transformed_encodings)

    def __getitem__(self, idx):
        return self.transformed_encodings[idx], self.encodings[idx]