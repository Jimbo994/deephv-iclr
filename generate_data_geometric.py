import os

from typing import Callable, Optional

import numpy as np
import torch

from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning

from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

def sample_uniformly(ndim, maxsol=200):
    """
    Idea of sampling approach from deep sets hv-net paper:
    • Step 1: Randomly sample an integer num ∈ [1, 100] where num denotes the number of solutions in the
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
        max_x (int): Maximum number of solutions in Pareto front
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
    def __init__(self, root: str, dataset_name: str, num_dim: int, max_x: int, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.dataset_name = dataset_name
        self.num_dim = num_dim
        self.max_x = max_x
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return self.dataset_name

    def process(self):
        data_list = []
        for i in tqdm(range(NUM_DATAPOINTS)):
            x = sample_uniformly(self.num_dim, maxsol=self.max_x)
            ref_point = torch.zeros(self.num_dim)
            bd = DominatedPartitioning(ref_point=ref_point, Y=x)
            volume = bd.compute_hypervolume()
            # if volume < 1e-5:
            #    logvolume = torch.tensor(1e-5)
            log_volume = np.log(volume)

            data = Data(x=x.flatten(), y=volume, y_log=log_volume, idx=i, N=len(x), M=self.num_dim)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

NUM_OBJECTIVES = 2
MAX_X = 200
NUM_DATAPOINTS = 100000

if __name__ == "__main__":
    path = os.getcwd()
    front = sample_uniformly(2, maxsol=200)
    print(front)
    #dataset = HV_dataset_uniform(path, 'dataset_uniform10_100_part1.pt', 10, 100)
