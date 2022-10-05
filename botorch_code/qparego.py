import os
import torch
import numpy as np
import botorch
import matplotlib.pyplot as plt
from botorch.acquisition import GenericMCObjective, qExpectedImprovement
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from get_problem import get_problem
from gpytorch.mlls import SumMarginalLogLikelihood
from utils import generate_initial_data, initialize_simple_model
import time

from botorch import fit_gpytorch_model

from botorch.utils.transforms import unnormalize, normalize

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


from pymoo.visualization.scatter import Scatter

import warnings
from botorch.exceptions import BadInitialCandidatesWarning

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x, prob.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
                                    )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qparego_and_get_observation(model, train_x, train_obj, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, prob.bounds)
    with torch.no_grad():
        pred = model.posterior(train_x).mean

    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(prob.num_objectives, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        acq_func = qExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            best_f=objective(train_obj).max(),
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=prob.bounds)
    new_obj_true = prob(new_x)
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
MC_SAMPLES = 128

master_path = "../botorch_results/parego/"
if not os.path.exists(master_path):
    os.makedirs(master_path)

verbose = True

problems = ['dtlz1', 'dtlz2', 'dtlz5', 'dtlz7', 'vehiclesafety']
dims = [3,4,5]

for problem in problems:
    if problem == 'vehiclesafety':
        dims = [3]
    else:
        dims = [3, 4, 5]
    for dim in dims:
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

                sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

                # optimize acquisition functions and get new observations
                new_x, new_obj, new_obj_true = optimize_qparego_and_get_observation(
                    model, train_x, train_obj, sampler
                )


                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                train_obj_true = torch.cat([train_obj_true, new_obj_true])

                if iteration % 10 == 0:
                    dom = is_non_dominated(train_obj_true)
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
                        f"\nBatch {iteration:>2}: Hypervolume (qparego) = "
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
