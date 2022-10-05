import os
import torch
import numpy as np
import botorch
import matplotlib.pyplot as plt

from torch.nn.functional import pad

import time
from get_problem import get_problem
from utils import generate_initial_data, initialize_simple_model

from botorch.utils.multi_objective import infer_reference_point
from gpytorch import ExactMarginalLogLikelihood
from pymoo.visualization.scatter import Scatter
from torch_geometric.data import Data

from botorch import fit_gpytorch_model

from botorch.utils.transforms import unnormalize, normalize

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from custom_acquisition_functions import BaselineDeepHVEHVIBatched, BaselineDeepHVEHVIBatchedPadded

import warnings
from botorch.exceptions import BadInitialCandidatesWarning

from models import DoubleDeepSetModel, DoubleDeepSetModelBatched

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

### HV Net imports
from torchmetrics.functional import accuracy, mean_absolute_error, mean_squared_error
#from hv_net.main_w_scalingequivariance import DoubleDeepSetModel, DataLoader, DoubleDeepSetLayer
#from hv_net.generate_data_geometric import HV_dataset
def transform(train_obj, ref_point):
    # subtract reference point
    train_obj_shifted = train_obj - ref_point
    # check if no points are worse than ref point
    mask = torch.any(train_obj_shifted < 0, dim=1)
    return train_obj_shifted[~mask]

def transform_mask_non_dominated(train_obj, ref_point):
    # subtract reference point
    train_obj_shifted = train_obj - ref_point
    # check if no points are worse than ref point
    mask = torch.any(train_obj_shifted < 0, dim=-1, keepdim=True)
    # set all points worse than ref points to zero
    masked = train_obj_shifted * ~mask
    # set all points that are dominated to zero.
    nondom = is_non_dominated(masked).unsqueeze(-1)
    return masked * nondom

def transform_mask_non_dominated_allmode(train_obj, ref_point):
    # subtract reference point
    train_obj_shifted = train_obj - ref_point
    shape = train_obj_shifted.shape
    # check if no points are worse than ref point
    mask = torch.any(train_obj_shifted < 0, dim=-1, keepdim=True)

    # pad train_obj_shifted in M dimension
    train_obj_shifted_pad = pad(train_obj_shifted, (0, 10-shape[1], 0, 0), "constant", 1)

    # set all points worse than ref points to zero
    masked = train_obj_shifted * ~mask
    masked_pad = train_obj_shifted_pad * ~mask
    # set all points that are dominated to zero.
    nondom = is_non_dominated(masked).unsqueeze(-1)
    return masked_pad * nondom

def optimize_hvi_and_get_observation(model, train_x, train_obj, hv_model):
    """
    What should be done here is to create a model for the Pareto Front and then find the y to target
    Then create scalarized objective that will target that y
    Then optimize and get candidate

    :param model:
    :param train_x:
    :param train_obj:
    :param sampler:
    :return: candidate point to evaluate next

    """
    # Now use this GP to find best objective values to target optimization on
    # Set previous best hypervolume on normalized objectives
    # We need to transform the objectives so that the ref_point is at 0. and no points are worse than the ref point.

    if MODE == 'all':
        train_obj_shifted = transform_mask_non_dominated_allmode(train_obj, prob.ref_point)
    else:
        train_obj_shifted = transform_mask_non_dominated(train_obj, prob.ref_point)

    # bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
    # volume_true = bd.compute_hypervolume()

    shape = train_obj_shifted.shape
    if shape[0] == 0 or train_obj_shifted.sum() == torch.tensor(0, **tkwargs):
        volume = torch.tensor(0., **tkwargs)
    else:
        # now pad tensor
        train_obj_shifted = pad(train_obj_shifted, (0, 0, 0, 100 - shape[0]), "constant", 0)
        shape = train_obj_shifted.shape
        data = Data(x=train_obj_shifted.flatten().type(torch.float), N=[shape[0]], M=[shape[1]], ptr=[0, shape[0] * shape[1]])
        with torch.no_grad():
            volume = hv_model(data).squeeze()

    # Now create objective, which measures hypervolume difference based on mean of GP
    # define acq func.
    # Note: beta is not being used

    if MODE == 'all':
        UCB = BaselineDeepHVEHVIBatchedPadded(model, hv_model, volume, train_obj, prob.ref_point, beta=0.3)
    else:
        UCB = BaselineDeepHVEHVIBatched(model, hv_model, volume, train_obj, prob.ref_point, beta=0.3)
    # Optimize acq func.
    acq_bounds = standard_bounds
    candidate, acq_value = optimize_acqf(
        UCB, bounds=acq_bounds, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )

    new_x = unnormalize(candidate.detach(), prob.bounds)
    new_obj_true = prob(new_x)
    print('new_y', new_obj_true)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}

N_TRIALS = 5
N_BATCH = 150
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 64
MODE = 'all'

master_path = "../botorch_results/deephv-ehvi-allmodel128/"
if not os.path.exists(master_path):
    os.makedirs(master_path)

verbose = True

problems = ['dtlz1', 'dtlz2', 'dtlz5', 'dtlz7', 'vehiclesafety']
dims = [3,4,5]

modelpath = '../models/'
# On Lisa
# modelpath =
channels = 128

for problem in problems:
    if problem == 'vehiclesafety':
        dims = [3]
    else:
        dims = [3,4,5]
    for dim in dims:
        if MODE == 'all':
            modelname = 'Mallpretzel_' + str(channels) + 'channels.ckpt'
            hv_model = DoubleDeepSetModelBatched()
            hv_model = hv_model.load_from_checkpoint(modelpath+modelname)
        else:
            modelname = 'M' + str(dim) +'_' +str(channels) + 'channels.ckpt'
            hv_model = DoubleDeepSetModelBatched()
            hv_model = hv_model.load_from_checkpoint(modelpath+modelname)
        # Set if you want to incorporate observational noise, here we omit it.
        NOISE_SE = torch.zeros(dim, **tkwargs)
        print(problem)
        # negate = true is required if problems are minimization problems, as botorch assumes maximization.
        if problem == 'vehiclesafety':
            prob = get_problem(problem, negate=True).to(**tkwargs)
        else:
            prob = get_problem(problem, dim=dim*2, num_objectives=dim, negate=True).to(**tkwargs)

        standard_bounds = torch.zeros(2, prob.dim, **tkwargs)
        standard_bounds[1] = 1

        #Set path to save experiments to:
        SAVE_PATH = master_path + problem + "_M" + str(dim) + "/"
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
            os.makedirs(SAVE_PATH + 'images')
        hvs_all = []
        times_all = []

        # average over multiple trials
        print('BOUNDS: ', prob.ref_point)
        for trial in range(1, N_TRIALS + 1):
            torch.manual_seed(trial)

            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            hvs = []
            times = []

            # call helper functions to generate initial training data and initialize model
            train_x, train_obj, train_obj_true = generate_initial_data(n= 2 * (prob.dim + 1), problem=prob, noise_se=NOISE_SE) # Sobol sampling
            mll, model = initialize_simple_model(train_x, train_obj, prob) # Standardization is handled in model

            # compute hypervolume
            bd = FastNondominatedPartitioning(ref_point=prob.ref_point, Y=train_obj_true)
            volume = bd.compute_hypervolume().item()

            hvs.append(volume)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, N_BATCH + 1):
                print(f"\nIteration {iteration:>2} of {N_BATCH} ", end="\n")

                t0 = time.time()

                # fit the models
                fit_gpytorch_model(mll)

                # optimize acquisition functions and get new observations
                new_x, new_obj, new_obj_true = optimize_hvi_and_get_observation(
                    model, train_x, train_obj, hv_model
                )

                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                train_obj_true = torch.cat([train_obj_true, new_obj_true])

                if iteration % 10 == 0:
                    dom = is_non_dominated(train_obj_true)
                    # Scatter(angle=(45, 45)).add(train_obj_true[dom].numpy()).show()
                    Scatter(angle=(45, 45)).add(train_obj_true[dom].numpy()).save(SAVE_PATH+'images/Front_rep' + str(trial) + '_' +str(iteration))

                # Reinitialize the models so they are ready for fitting on next iteration
                # Note: Botorch developers find improved performance from not warm starting the model hyperparameters
                # using the hyperparameters from the previous iteration
                mll, model = initialize_simple_model(train_x, train_obj, prob)

                t1 = time.time()
                times.append(t1-t0)

                bd = DominatedPartitioning(ref_point=prob.ref_point, Y=train_obj_true)
                volume_true = bd.compute_hypervolume().item()
                hvs.append(volume_true)

                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: Hypervolume (deephv-ehvi) = "
                        f"({hvs[-1]:>4.2f}), "
                        f"time = {t1 - t0:>4.2f}.", end=""
                    )
                else:
                    print(".", end="")

            hvs_all.append(hvs)
            times_all.append(times)

            # save train_obj
            np.savetxt(SAVE_PATH + 'train_obj_' + str(trial) + '.txt', train_obj)

            # save train_x
            np.savetxt(SAVE_PATH +'train_x_' + str(trial) + '.txt', train_x)

            # Save files (probably better to save every iteration
            # Save hypervolumes
            np.savetxt(SAVE_PATH +'hvs_all.txt', hvs_all)
            np.savetxt(SAVE_PATH +'times_all.txt', times_all)