import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError, R2Score, SpearmanCorrCoef
import numpy as np

class DiffDataset(Dataset):
    def __init__(self, embedding_dictionary, data_split):
        self.embedding_dictionary = embedding_dictionary

        data = np.loadtxt(data_split, delimiter=",", dtype=str)

        self.ab_1_ids = data[:, 0]
        self.ab_2_ids = data[:, 1]
        self.differences = torch.tensor(data[:, 2].astype(np.float32))

        self.ab_1_embeddings = torch.stack([torch.tensor(embedding_dictionary[ab]) for ab in self.ab_1_ids])
        self.ab_2_embeddings = torch.stack([torch.tensor(embedding_dictionary[ab]) for ab in self.ab_2_ids])

    def __len__(self):
        return len(self.differences)

    def __getitem__(self, index):
        return self.ab_1_embeddings[index], self.ab_2_embeddings[index], self.differences[index]

def get_data_loader(embedding_dictionary, data_split, batch_size, shuffle):
    dataset = DiffDataset(embedding_dictionary, data_split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def step(model, data_loader, device, criterion, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)
    spearman_metric = SpearmanCorrCoef().to(device)
    all_ys, all_predictions = [], []
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for ab_1, ab_2, ys in data_loader:
            ab_1, ab_2, ys = ab_1.to(device), ab_2.to(device), ys.to(device)
            
            if train:
                optimizer.zero_grad()

            predictions = model(ab_1, ab_2)
            loss = criterion(predictions, ys)

            batch_size = len(ab_1)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if train:
                loss.backward()
                optimizer.step()
            
            all_ys.append(ys)
            all_predictions.append(predictions)

    avg_loss = total_loss / total_samples
    all_ys = torch.cat(all_ys, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    mae = mae_metric(all_predictions, all_ys)
    r2 = r2_metric(all_predictions, all_ys)
    spearman = spearman_metric(all_predictions, all_ys)
    return avg_loss, mae.item(), r2.item(), spearman.item()
