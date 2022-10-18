from pymoo_code.sms_adjusted import SMSEMOA, LeastHypervolumeContributionSurvival
from pymoo.problems import get_problem
import torch
import numpy as np
import time

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions

from models import DoubleDeepSetModelBatched, DoubleDeepSetModel
from torch_geometric.data import Data

import pandas as pd

import os

def transform(x, ref_point, do_normalize=True):
    """
    Transforms problem from minimization to maximization and vice versa.
    :param x:
    :param ref_point:
    :return:
    """
    x = -x
    ref_point = - ref_point
    x = x - ref_point

    # Check if any points are worse than ref points and remove them.
    mask = np.any(x.numpy(), where=x.numpy() < 0, axis=1)
    return x[~mask]

if __name__ == "__main__":
    repetitions = 5
    # set what experiments you want to run.
    run_pareto = False # Generate example Pareto Fronts for each problem
    run_baseline = False
    run_deephv = True
    run_nsga = False
    n_generations = 10
    problems = ['dtlz1', 'dtlz2', 'convex_dtlz2', 'dtlz5', 'dtlz7']#, 'wfg1', 'wfg2', 'wfg3']
    dims = [3,4,5,6,7,8,9,10]
    channels = ['allpretzel', 'all', 128, 256]
    # modes = ['normal', 'all']
    # Partitions required to roughly pick at least 100 points on Pareto front, typically more though
    pareto_front_partitions = {3:13, 4:7, 5:5, 6:4, 7:3, 8:3, 9:3, 10:3}
    path = os.getcwd()
    model_path = os.path.join(path, 'models')
    save_path = 'pymoo_results_deephv_pretzel128/'

    # Run Pareto front generation
    if run_pareto:
        for problem in problems:
            os.makedirs(save_path + problem + '/images', exist_ok=True)
            for dim in dims:
                print('Generating Pareto Fronts for problem {} with M={}'.format(problem, str(dim)))
                prob = get_problem(problem, n_var=2 * dim, n_obj=dim)
                points = get_reference_directions("uniform", dim, n_partitions=pareto_front_partitions[dim])
                if problem == 'dtlz5' or problem == 'dtlz7':
                    try:
                        pf = prob.pareto_front()
                        Scatter(angle=(45, 45)).add(pf).save(
                            save_path + problem + '/images/ParetoFront_' + str(len(pf)) + '_' + str(problem) + '_' + str(dim))

                    except:
                        print('Pareto front prediction not available')
                else:
                    pf = prob.pareto_front(points)
                    Scatter(angle=(45, 45)).add(pf).save(
                        save_path + problem + '/images/ParetoFront_' + str(len(pf)) + '_' + str(problem) + '_' + str(dim))

                # Plot Pareto front and save

    # Run SMS-EMOA with DeepHV model and Exact HV.
    for problem in problems:
        os.makedirs(save_path + problem + '/images', exist_ok=True)
        for dim in dims:
            # Probably open DataFrame here and save repeats in there?
            results_df_deephv = pd.DataFrame(columns=['repeat', 'n_gen', 'time', 'n_eval', 'n_nds', 'hv', 'hv_model', 'channels'])
            fronts_df_deephv = pd.DataFrame(columns=['front', 'channels'])
            results_df_baseline = pd.DataFrame(columns=['repeat', 'n_gen', 'time', 'n_eval', 'n_nds', 'hv', 'hv_model'])
            fronts_df_baseline = pd.DataFrame(columns=['front'])

            results_df_nsga2 = pd.DataFrame(columns=['repeat', 'n_gen', 'time', 'n_eval', 'n_nds', 'hv', 'hv_model'])
            fronts_df_nsga2 = pd.DataFrame(columns=['front'])

            # Initiate problem with 2*dim input variables, dim objectives
            prob = get_problem(problem, n_var=2 * dim, n_obj=dim)

            ref_point_pymoo = np.ones(dim)
            if problem == 'dtlz7':
                ref_point_pymoo = np.ones(dim)*15
            if problem == 'dtlz1':
                ref_point_pymoo = np.ones(dim)*400


            ind_pymoo = HV(ref_point=ref_point_pymoo)
            ref_point_deephv = np.zeros(dim)
            for rep in range(repetitions):
                if run_baseline:
                    print('Running exact hv sms-emoa calculations for problem {} with M={} for repeat {} out of {}'
                          .format(problem, str(dim), str(rep+1), str(repetitions)))
                    # Run SMS-EMOA with exact Hypervolume
                    # eps is set so that reference point [1]^dim is not shifted. This is key for the conversion
                    # to maximization for DeepHV
                    algorithm = SMSEMOA(pop_size=100, survival=LeastHypervolumeContributionSurvival(eps=0.0))
                    algorithm.setup(prob, termination=('n_gen', n_generations), seed=rep, verbose=False)
                    tot_time = 0
                    while algorithm.has_next():

                        start = time.time()
                        # do the next iteration
                        algorithm.next()
                        # Save every intermediate
                        end = time.time()
                        tot_time += end-start

                        front = algorithm.result().F
                        # Compute HV with pymoo_code
                        hv_pymoo = ind_pymoo(front)

                        # Save data to DataFrame
                        values_to_add = {'repeat': rep, 'n_gen': algorithm.n_gen, 'time': end-start, #This should be tot_time!!
                                         'n_eval': algorithm.evaluator.n_eval, 'n_nds': len(front),
                                            'hv': hv_pymoo, 'hv_model': 0}

                        front_to_add = {'front': front}

                        row_to_add = pd.Series(values_to_add)
                        row_to_add_front = pd.Series(front_to_add)

                        results_df_baseline = results_df_baseline.append(row_to_add, ignore_index=True)
                        results_df_baseline.to_csv(save_path + problem +'/results_exact_hv' + '_' + str(problem)+ '_dim' + str(dim)+'.csv')

                        fronts_df_baseline = fronts_df_baseline.append(row_to_add_front, ignore_index=True)
                        fronts_df_baseline.to_csv(save_path + problem +'/fronts_exact_hv' + '_' + str(problem)+ '_dim' + str(dim)+'.csv')


                        # Save plot of front
                        Scatter(angle=(45, 45)).add(front).save(
                            save_path + problem +'/images/ExactHVFront_ngen' + str(algorithm.n_gen) + '_rep' + str(rep) + '_' + str(problem) + '_dim' + str(dim))

                if run_nsga:
                    print('Running exact nsga-II calculations for problem {} with M={} for repeat {} out of {}'
                          .format(problem, str(dim), str(rep+1), str(repetitions)))
                    # Run SMS-EMOA with exact Hypervolume
                    # eps is set so that reference point [1]^dim is not shifted. This is key for the conversion
                    # to maximization for DeepHV
                    algorithm = NSGA2(pop_size=100)
                    algorithm.setup(prob, termination=('n_gen', n_generations), seed=rep, verbose=False)
                    tot_time = 0
                    while algorithm.has_next():

                        start = time.time()
                        # do the next iteration
                        algorithm.next()
                        # Save every intermediate
                        end = time.time()
                        tot_time += end-start

                        front = algorithm.result().F
                        # Compute HV with pymoo_code
                        hv_pymoo = ind_pymoo(front)

                        # Save data to DataFrame
                        values_to_add = {'repeat': rep, 'n_gen': algorithm.n_gen, 'time': end-start, # This should be tot_time
                                         'n_eval': algorithm.evaluator.n_eval, 'n_nds': len(front),
                                            'hv': hv_pymoo, 'hv_model': 0}

                        front_to_add = {'front': front}

                        row_to_add = pd.Series(values_to_add)
                        row_to_add_front = pd.Series(front_to_add)

                        results_df_nsga2 = results_df_nsga2.append(row_to_add, ignore_index=True)
                        results_df_nsga2.to_csv(save_path + problem +'/results_nsga2' + '_' + str(problem)+ '_dim' + str(dim)+'.csv')

                        fronts_df_nsga2 = fronts_df_nsga2.append(row_to_add_front, ignore_index=True)
                        fronts_df_nsga2.to_csv(save_path + problem +'/fronts_nsga2' + '_' + str(problem)+ '_dim' + str(dim)+'.csv')


                        # Save plot of front
                        Scatter(angle=(45, 45)).add(front).save(
                            save_path + problem +'/images/NSGA2Front_ngen' + str(algorithm.n_gen) + '_rep' + str(rep) + '_' + str(problem) + '_dim' + str(dim))

                # Save every intermediate
                if run_deephv:
                    #Run SMS-EMOA with DeepHV Hypervolume
                    if dim in [3,5,8,10]:
                        channels = ['allpretzel']#, 'all', 128, 256]
                    else:
                        channels = ['allpretzel']#, 'all', 128, 256]
                    for channel in channels:
                        print('Running deephv sms-emoa calculations for problem {} with M={} using model with {} channels for repeat {} out of {}'
                              .format(problem, str(dim), str(channel), str(rep + 1), str(repetitions)))
                        # Load model
                        if channel == 'all' or channel == 'allpretzel':
                            #channel = 256
                            modelname = 'M' + channel + '_' + str(128) + 'channels.ckpt'
                            model = DoubleDeepSetModelBatched().load_from_checkpoint(
                                checkpoint_path=os.path.join(model_path, modelname)
                            )
                        else:
                            modelname = 'M' + str(dim) + '_' + str(channel) + 'channels.ckpt'
                            model = DoubleDeepSetModelBatched().load_from_checkpoint(
                                checkpoint_path=os.path.join(model_path, modelname)
                        )
                        # Run SMS-EMOA with DeepHV Hypervolume
                        # eps is set so that reference point [1]^dim is not shifted. This is key for the conversion
                        # to maximization for DeepHV
                        algorithm = SMSEMOA(pop_size=100, survival=LeastHypervolumeContributionSurvival(eps=0.0, model=model))
                        algorithm.setup(prob, termination=('n_gen', n_generations), seed=rep, verbose=False)
                        tot_time = 0

                        while algorithm.has_next():
                            start = time.time()
                            # do the next iteration
                            algorithm.next()
                            # Save every intermediate time, only counting actual time SMS-EMOA is running
                            end = time.time()
                            tot_time += end - start

                            front = algorithm.result().F
                            # Compute HV with pymoo_code
                            hv_pymoo = ind_pymoo(front)

                            # Compute HV with model
                            front_transform = transform(torch.tensor(front), ref_point_pymoo)
                            shape = np.shape(front_transform)

                            # create data format suitable for model
                            if channel=='all' or channel =='allpretzel':
                                # padded with 1s and zeros
                                # First pad 1s
                                front_transform_padded = np.pad(front_transform, ((0, 0), (0, 10 - shape[1])), 'constant',
                                                                constant_values=((0., 0.), (0.,1.)))
                                # Pad remainder with zeros
                                front_transform_padded = np.pad(front_transform_padded, ((0, 100 - shape[0]), (0, 0)), 'constant',
                                                                constant_values=((0., 0.),(0.,0.)))
                                data = Data(x=torch.tensor(front_transform_padded.flatten()).type(torch.float), N=[100],
                                                   M=[10], ptr=[0, 100 * 10])
                            else:
                                # Pad remainder with zeros
                                front_transform_padded = np.pad(front_transform, ((0, 100 - shape[0]), (0, 0)), 'constant',
                                                                constant_values=((0., 0.),(0.,0.)))
                                data = Data(x=torch.tensor(front_transform_padded.flatten()).type(torch.float), N=[100],
                                                   M=[shape[1]], ptr=[0, 100 * shape[1]])
                                # front_transform_padded = np.pad(front_transform, ((0, 100-shape[0]), (0, 0)), 'constant', constant_values=(0.,0.))
                                # data_padded = Data(x=torch.tensor(front_transform_padded.flatten()).type(torch.float), N=[100], M=[shape[1]], ptr=[0, 100*shape[1]])
                                # data = Data(x=torch.tensor(front_transform.flatten()).type(torch.float), N=[shape[0]],
                                #             M=[shape[1]], ptr=[0, shape[0] * shape[1]])

                            with torch.no_grad():
                                if data.N[0] == 0:
                                    hv_model = 0.0
                                else:
                                    if channel=='all':
                                        hv_model = model(data).numpy()[0][0][0]
                                    else:
                                        hv_model = model(data).numpy()[0][0][0]

                            # Save data to DataFrame
                            values_to_add = {'repeat': rep, 'n_gen': algorithm.n_gen, 'time': end - start,
                                             'n_eval': algorithm.evaluator.n_eval,
                                             'n_nds': len(front),
                                             'hv': hv_pymoo, 'hv_model': hv_model, 'channels' : channel}

                            front_to_add = {'front': front, 'channels': channel}

                            row_to_add = pd.Series(values_to_add)
                            row_to_add_front = pd.Series(front_to_add)

                            results_df_deephv = results_df_deephv.append(row_to_add, ignore_index=True)
                            results_df_deephv.to_csv(
                                save_path + problem + '/results_deep_hv_batched' + '_' + str(problem) + '_dim' + str(
                                    dim) + '.csv')

                            fronts_df_deephv = fronts_df_deephv.append(row_to_add_front, ignore_index=True)
                            fronts_df_deephv.to_csv(
                                save_path + problem + '/fronts_deep_hv_batched' + '_' + str(problem) + '_dim' + str(
                                    dim) + '.csv')

                            # Save plot of front
                            Scatter(angle=(45, 45)).add(front).save(
                                save_path + problem + '/images/DeepHVFront_ngen' + str(algorithm.n_gen) + '_rep' + str(
                                    rep) + '_' + str(problem) + '_dim' + str(dim) + '_channels' + str(channel))
