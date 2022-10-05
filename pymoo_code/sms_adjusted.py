import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.indicators.hv import hvc_looped
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from pymoo.indicators.hv.exact import hv_exact

import torch
from torch.nn.functional import pad
from torch_geometric.data import Data

import os

from torch_geometric.loader import DataLoader

from models import DoubleDeepSetModel

path = os.getcwd()
# ---------------------------------------------------------------------------------------------------------
# Environmental Survival - Remove the solution with the least HV contribution
# ---------------------------------------------------------------------------------------------------------


def transform(x, ref_point, do_normalize=True):
    """
    :param x:
    :param ref_point:
    :return:
    """
    # if normalize:
    #     min = torch.tensor(x.min(axis=0))-1e-3
    #     max = torch.tensor(x.max(axis=0))
    #     x = torch.tensor(x)
    #     x = normalize(x, torch.stack([min, max]))
    x = -x
    ref_point = - ref_point
    x = x - ref_point
    #if normalize:
    #    x = unnormalize(x, torch.stack([min, max]))
    return x

def hv_deephv(ref_point, F):
    """
    F should already be normalized here.
    :param ref_point:
    :param F:
    :return:
    """
    # model_path = os.path.join(path, 'models')
    # dim = 3
    # channel = 128
    #
    # modelname = 'M' + str(dim) + '_' + str(channel) + 'channels.ckpt'
    #
    # model = DoubleDeepSetModel().load_from_checkpoint(
    #     checkpoint_path=os.path.join(model_path, modelname)
    # )

    F = torch.tensor(F)
    F_t = transform(F, ref_point)
    shape = F_t.shape
    if shape[0] == 0:
        return 0.
    data = Data(x=torch.tensor(F_t.flatten()).type(torch.float), N=[shape[0]], M=[shape[1]],
                ptr=[0, shape[0] * shape[1]])
    with torch.no_grad():
        hv = MODEL(data)
    return float(hv.numpy()[0][0])


def hvc_looped_batched(ref_point, F, func):
    with torch.no_grad():
        hv = func(ref_point, F)
        hvc = []
        data_list = []


        if len(F) == 1:
            print('this happened')
            return np.array([hv])
        else:
            for k in range(len(F)):
                v = np.full(len(F), True)
                v[k] = False
                F_tensor = torch.tensor(F[v])
                F_t = transform(F_tensor, ref_point)
                shape = F_t.shape

                # Pad with 1s if we use 'all' model
                if MODEL.num_dim == 10:
                    # either we use the 'all' model or we use dim=10 model
                    # for the former we need to pad with 1s for the latter
                    # dim is already 10 so we won't pad.
                    F_t = pad(F_t, (0, 10 - shape[1], 0, 0), "constant", 1)

                # pad with zeros
                F_t = pad(F_t, (0,0,0, 100-shape[0]), "constant", 0)
                shape = F_t.shape

                data = Data(x=torch.tensor(F_t.flatten()).type(torch.float), N=shape[0], M=shape[1],
                        ptr=[0, shape[0] * shape[1]])
                data_list.append(data)

        loader = DataLoader(data_list, batch_size=len(data_list))
        for batch in loader:
            hvs = MODEL(batch)
    return (hv - hvs.flatten()).numpy()


def hvc_deephv_loopwise(ref_point, F):
    return hvc_looped_batched(ref_point, F, hv_deephv)

class LeastHypervolumeContributionSurvival(Survival):

    def __init__(self, eps=10.0, model=None) -> None:
        super().__init__(filter_infeasible=True)
        self.eps = eps
        self.model = model

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # get the actual front as individuals
            front = pop[front]
            front.set("rank", k)

            if len(survivors) + len(front) > n_survive:

                # normalize all the function values for the front
                F = front.get("F")
                # Jim: ideal and nadir come from the entire population, so normalized F is not always on the bounds [0,1]
                # But in the extreme case it is at [0,1]
                #
                #
                F = normalize(F, ideal, nadir+0.1)

                # define the reference point and shift it a bit - F is already normalized!
                ref_point = np.full(problem.n_obj, 1.0 + self.eps)

                _, n_obj = F.shape

                # choose the suitable hypervolume method
                clazz = ExactHypervolume
                #clazz2 = ExactHypervolume
                if n_obj == 2:
                    clazz = ExactHypervolume2D
                elif n_obj > 5 and self.model == None:
                    clazz = ApproximateMonteCarloHypervolume

                # finally do the computation
                if self.model == None:
                    hv = clazz(ref_point, func_hv=hv_exact).add(F)
                else:
                    # define model here globally (lazy) but means we don't have to pass it along to all functions
                    # and can leave pymoo_code code mostly intact.
                    global MODEL
                    MODEL = self.model
                    hv = clazz(ref_point, func_hv=hv_deephv, func_hvc=hvc_deephv_loopwise).add(F)
                    #hv2 = clazz2(ref_point).add(F)
                #print('HERE', hv.hv, hv2.hv)

                # current front sorted by crowding distance if splitting
                while len(survivors) + len(front) > n_survive:
                    k = hv.hvc.argmin()
                    hv.delete(k)
                    front = np.delete(front, k)

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return Population.create(*survivors)


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament
# ---------------------------------------------------------------------------------------------------------


def cv_and_dom_tournament(pop, P, *args, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            # if one dominates another choose the nds one
            rel = Dominator.get_relation(a_f, b_f)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------

class SMSEMOA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=cv_and_dom_tournament),
                 crossover=SBX(prob_exch=0.5),
                 mutation=PM(),
                 survival=LeastHypervolumeContributionSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 normalize=True,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.normalize = normalize
    def _advance(self, infills=None, **kwargs):

        ideal, nadir = None, None

        # estimate ideal and nadir from the current population (more robust then from doing it from merged)
        #if self.normalize:
        #    F = self.pop.get("F")
        #    ideal, nadir = F.min(axis=0), F.max(axis=0) + 1e-32

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self,
                                    ideal=ideal, nadir=nadir, **kwargs)


parse_doc_string(SMSEMOA.__init__)
