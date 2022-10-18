from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from tqdm import tqdm

from generate_data_geometric import HV_dataset_uniform
import os
import torch.nn.functional as F
path = os.getcwd()

n_dim = 4
dataset_5obj_uniform = HV_dataset_uniform(root=path, dataset_name='data_uniform4_100_fixed.pt', num_dim=n_dim, max_x=100)  # 1M datapoints

MAX_X = 100
# dimension to pad to.
MAX_DIM = 10

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data

class PaddedDataset(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data_uniform' +str(n_dim) +'_100_padded_10_N100.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        ptr_start = 0
        for i in tqdm(range(len(self.dataset.data.idx))):
            N = self.dataset.data.N[i]
            M = self.dataset.data.M[i]
            ptr_end = ptr_start + (N * M)
            y = self.dataset.data.y[i]
            y_log = self.dataset.data.y_log[i]
            x = self.dataset.data.x[ptr_start: ptr_end]
            x = x.reshape(N, M)  # NxM
            # first pad the M dimension with ones to have M=6
            if i == 0:
                print(MAX_DIM - M)
            x_new = F.pad(x, (0, MAX_DIM - M), "constant", 1)
            # then pad the N dimensions to N = max_dim
            if N != MAX_X:
                zeros_tensor = torch.zeros(MAX_X - N, MAX_DIM) + torch.tensor(float('nan'))

            x_new = torch.cat((x_new, zeros_tensor))
            if i == 0:
                print(MAX_DIM - M)
                ref_point = torch.zeros(MAX_DIM)
                bd = DominatedPartitioning(ref_point=ref_point, Y=x_new)
                volume = bd.compute_hypervolume()
                print(volume, y)
            ptr_start = ptr_end

            data = Data(x=x_new.flatten(), y=y, y_log=y_log, idx=i, N=MAX_X,  M=MAX_DIM)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

data = PaddedDataset(root=path, dataset=dataset_4obj_uniform)