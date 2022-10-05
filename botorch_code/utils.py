import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize

def generate_initial_data(n, problem, noise_se):
    # generate training data
    train_x = draw_sobol_samples(
        bounds=problem.bounds,n=1, q=n, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise_se
    return train_x, train_obj, train_obj_true

def initialize_simple_model(train_x, train_obj, problem):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=problem.num_objectives))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model