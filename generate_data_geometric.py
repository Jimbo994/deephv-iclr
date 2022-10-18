import os

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning

from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

def sample_uniformly(ndim, maxsol=100):
    """
    Idea of sampling approach from deep sets hv-net paper:
    • Step 1: Randomly sample an integer num ∈ [1, maxsol] where num denotes the number of solutions in the
      solution set.
    • Step 2: Randomly sample 1000 solutions in [0, 1]^m as candidate solutions.
    • Step 3: Apply non-dominated sorting to these 1000 solutions and obtain different fronts {F1, F2, ...}
      where F1 is the first front (i.e., the set of nondominated solutions in the 1000 solutions) and F2 is
      the set of non-dominated solutions after all solutions in F1 are removed.
    • Step 4: Identify all the fronts Fi with |Fi| ≥ num. If no front satisfies this condition, go back to Step 2.
    • Step 5: Randomly pick one front Fi with |Fi| ≥ num and randomly select num solutions from this front
    to construct one solution set

    :param ndim: dimensionality of data you want to generate
    :param maxsol: maximum number of solutions in Pareto set
    :return: a pareto front
    """
    num_sols = 0
    accepted_front = []
    num = np.random.randint(low=1, high=maxsol, size=1)

    # keep going until we have at least one solution
    while num_sols == 0:
        sol = torch.rand((1000, ndim))
        nondom = is_non_dominated(sol)
        dom = ~nondom  # invert trues and falses

        if nondom.sum() > torch.tensor(num):
            accepted_front.append(sol[nondom])
            num_sols += 1
        sol = sol[dom]  # remove non-dominated points
        while nondom.sum() > 1 and len(sol) > num:
            nondom = is_non_dominated(sol)
            dom = ~nondom
            if nondom.sum() > torch.tensor(num):
                accepted_front.append(sol[nondom])
                num_sols += 1
            sol = sol[dom]

    # now pick random accepted pareto front, and reduce to length of num
    if len(accepted_front) > 1:
        idx = np.random.choice(len(accepted_front))
        front = accepted_front[idx]
    else:
        front = accepted_front[0]
    front = front[np.random.choice(len(front), size=num, replace=False)]
    return front

class HV_dataset_uniform(InMemoryDataset):
    """"
    HV_dataset loader or generator, generation happens by sampling using sample_uniformly() function.
        Args:
        root (string): Root directory where the dataset should be saved.
        dataset_name (string): name of what dataset should be loaded if it exists, or where it should be saved if not.
        num_dim (int): dimensions of problem.
        max_y (int): Maximum number of solutions in Pareto front
        max_dim (int): Maximum number of dimensions for padding
        num_datapoints (int): Number of datapoints to generate
        padding (bool): Indicating if the data needs to be padded up to a certain dimension M and number of solutions N
        sample_mode (str): indicating the sample method, currently accepts uniform and problem
        matlab (bool): if matlab (both installation and code) is available to speed up HV calculations
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root: str, dataset_name: str, num_dim: int, max_y: int,  max_dim: int,  num_datapoints: int, padding: bool = False,
                 sample_mode: str = 'uniform', matlab: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.dataset_name = dataset_name
        self.num_dim = num_dim
        self.max_y = max_y
        self.max_dim = max_dim
        self.num_datapoints = num_datapoints
        self.padding = padding
        self.sample_mode = sample_mode
        self.matlab = matlab
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return self.dataset_name

    def process(self):
        data_list = []
        problem = 'none'

        if self.matlab:
            # activate matlab engine
            import matlab.engine
            print('starting matlab engine')
            eng = matlab.engine.start_matlab()
            # change matlab directory to directory where code is placed
            eng.cd('matlab_code')

        for i in tqdm(range(self.num_datapoints)):
            if self.sample_mode == 'uniform':
                y = sample_uniformly(self.num_dim, maxsol=self.max_y)
            if self.matlab:
                # Matlab code is slightly different to BoTorch code, so we need to negate front.
                y_matlab = y * -1
                y_matlab = matlab.double(y_matlab.tolist())
                y_matlab = eng.reshape(y_matlab, matlab.double([1, y.shape[0], self.num_dim]))
                volume_matlab = eng.HV(y_matlab, matlab.double(0))
                volume = volume_matlab
                # To check if Matlab and BoTorch code is the same.
                # ref_point = torch.zeros(self.num_dim)
                # bd = DominatedPartitioning(ref_point=ref_point, Y=y)
                # volume_botorch = bd.compute_hypervolume()
                # print(volume_botorch, volume_matlab)
            else:
                ref_point = torch.zeros(self.num_dim)
                bd = DominatedPartitioning(ref_point=ref_point, Y=y)
                volume = bd.compute_hypervolume()
            log_volume = np.log(volume)
            initial_scale = y.abs().max(dim=0)[0]
            initial_scale_inv = 1 / initial_scale
            volume_normalized = volume * initial_scale_inv.prod()

            if self.padding:
                shape = y.shape  # M, N
                N, M = shape[0], shape[1]
                # first pad the M dimension with ones to have M=10
                y_new = F.pad(y, (0, self.max_dim - M), "constant", 1)
                # then pad the N dimensions to N = max_dim
                if N != self.max_y:
                    zeros_tensor = torch.zeros(self.max_y - N, self.max_dim)
                    y_new = torch.cat((y_new, zeros_tensor))
                shape = y.shape
                y = y_new
                M = self.max_dim
                data = Data(x=y.flatten(), y=volume, y_norm=volume_normalized, idx=i, N=len(y), N_true=shape[0], M=M,
                            M_true=self.num_dim, problem=problem)
            else:
                data = Data(x=y.flatten(), y=volume, y_norm=volume_normalized, idx=i, N=len(y), M=M,
                            M_true=self.num_dim, problem=problem)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

if __name__ == "__main__":
    path = os.getcwd()
    NUM_OBJECTIVES = 5
    MAX_SOL = 100
    MAX_DIM = 10
    NUM_DATAPOINTS = 100000

    dataset = HV_dataset_uniform(path, 'data_uniform5_100_padded_10_N100.pt', num_dim=NUM_OBJECTIVES, max_y=MAX_SOL, max_dim = MAX_DIM,
                                 num_datapoints = NUM_DATAPOINTS, padding=True, sample_mode='uniform', matlab=False)
