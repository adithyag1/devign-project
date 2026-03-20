from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
import torch


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    def get_loader(self, batch_size, shuffle=True):
        use_cuda = torch.cuda.is_available()
        num_workers = 2
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=use_cuda,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
        )
