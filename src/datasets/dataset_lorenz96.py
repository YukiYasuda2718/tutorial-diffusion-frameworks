import torch
import xarray as xr
from torch.utils.data import Dataset


class DatasetLorenz96(Dataset):
    def __init__(self, path_to_dataarray: str):

        self.data = xr.load_dataarray(path_to_dataarray)
        assert self.data.dims == ("batch", "time", "space")

        self.n_batch, self.n_times, self.n_spaces = self.data.shape
        self.mean = self.data.mean().item()
        self.std = self.data.std().item()

        self.dtype = torch.float32

    def __len__(self):
        return self.data.shape[0]  # batch dimension

    def standardize(self, data):
        return (data - self.mean) / self.std

    def standardize_inversely(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.data[idx].values  # time x space
        standardized = self.standardize(data)
        ret = torch.tensor(standardized, dtype=self.dtype)
        assert ret.shape == (self.n_times, self.n_spaces)
        return {"y0": ret}
