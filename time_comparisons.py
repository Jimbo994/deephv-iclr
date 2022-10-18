import torch
from torch_geometric.loader import DataLoader

from models import DoubleDeepSetModel
from generate_data_geometric import HV_dataset_uniform
import os
import time
import numpy as np

def transform(x, ref_point, ):
    """
    :param x:
    :param ref_point:
    :return:
    """
    x = -x
    ref_point = - ref_point
    x = x - ref_point
    return x

path = os.getcwd()
model_path = os.path.join(path, 'models')
savepath = os.path.join(path,'results')

repeats = 3
num_val = 100
batch_size = num_val

all_times_model = []
all_times_pymoo_mc = []
all_times_botorch = []
all_times_pymoo = []

dims = [3, 4, 5, 6, 7 , 8, 9]#,7,8,9,10]
channel = 256

for dim in dims:
    print('Busy with dimension: ', dim)
    # load dataset and loaders here
    if dim > 6:
        ds_name = 'dataset_uniform' + str(dim) + '_100_matlab.pt'
    else:
        ds_name = 'dataset_uniform' + str(dim) + '_100_final.pt'
    dataset = HV_dataset_uniform(root=path, dataset_name=ds_name,
                                 num_dim=dim, max_x=100).shuffle() # to replicate  splits that were used for training

    dataset.data.x = dataset.data.x.nan_to_num(0.)
    num_samples = len(dataset)

    val_dataset = dataset[:num_val]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    data = next(iter(val_loader))

    # load model
    modelname = 'M' + str(dim) + '_' + str(channel) + 'channels.ckpt'
    model = DoubleDeepSetModel().load_from_checkpoint(
        checkpoint_path=os.path.join(model_path, modelname)
    )
    model = model.load_from_checkpoint(
        checkpoint_path=os.path.join(model_path, modelname)
    )

    time.sleep(2)

    times_model = []
    for i in range(repeats):
        start = time.time()
        with torch.no_grad():
            # this is now evaluated in batch which is a bit unfair.
            preds = model(data)
        end = time.time()
        times_model.append(end-start)

    print(times_model)
    all_times_model.append(times_model)

    times_botorch = []
    for j in range(repeats):
        t = 0
        for i in range(len(ptr)-1):
            y = x[ptr[i]:ptr[i+1]]
            y = y.reshape(N[i], M[i])
            # only measure time of actual HV compute
            start = time.time()
            bd = DominatedPartitioning(ref_point=torch.zeros(M[i]), Y=y)
            volume = bd.compute_hypervolume()
            end = time.time()
            t += end-start
            # print(volume, hvs[i])
        print('BOTORCH done with repeat:', j, 'in', t, 'seconds')
        times_botorch.append(t)
    all_times_botorch.append(times_botorch)

    ref_point = np.ones(dim)

    times_pymoo = []
    for j in range(repeats):
        t = 0
        for i in range(len(ptr)-1):
            y = x[ptr[i]:ptr[i+1]]
            y = y.reshape(N[i], M[i])
            y_pymoo = transform(y, ref_point).numpy()
            # only measure time of actual HV compute
            start = time.time()
            volume = hv_exact(ref_point, y_pymoo)
            end = time.time()
            t += end-start
            #print(volume, hvs[i])
        print('PYMOO done with repeat:', j, 'in', t, 'seconds')
        times_pymoo.append(t)
    all_times_pymoo.append(times_pymoo)

    times_pymoo_mc = []
    for j in range(repeats):
        t = 0
        for i in range(len(ptr)-1):
            y = x[ptr[i]:ptr[i+1]]
            y = y.reshape(N[i], M[i])
            y_pymoo = transform(y, ref_point).numpy()
            clazz = ApproximateMonteCarloHypervolume
            # only measure time of actual HV compute
            start = time.time()
            volume = clazz(ref_point)._calc(ref_point, y_pymoo)
            end = time.time()
            t += end-start
            #print(volume, hvs[i])
        print('PYMOO MC done with repeat:', j, 'in', t, 'seconds')
        times_pymoo_mc.append(t)
    all_times_pymoo_mc.append(times_pymoo_mc)

    np.savetxt(os.path.join(savepath, 'all_times_model' + str(channel) +'.txt'), all_times_model)
    np.savetxt(os.path.join(savepath, 'all_times_pymoo_mc.txt'), all_times_pymoo_mc)
    np.savetxt(os.path.join(savepath, 'all_times_pymoo.txt'), all_times_pymoo)
    np.savetxt(os.path.join(savepath, 'all_times_botorch.txt'), all_times_botorch)
