import os

import wandb

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torch_geometric.loader import DataLoader

from models import DoubleDeepSetModelBatched
from generate_data_geometric import HV_dataset_uniform

import sys

pytorch_lightning.seed_everything(42)

def main():
    if len(sys.argv) == 1:
        dim = 4
        inp_channels = 128
    else:
        dim = int(sys.argv[1])
        inp_channels = int(sys.argv[2])

    path = os.getcwd()
    #print(dim, inp_channels, type(dim))
    # always check what is in the HV code
    dataset = HV_dataset_uniform(path, 'data_uniform5_100_padded_10_N100.pt', num_dim=5, max_y=100, max_dim = 10,
                                 num_datapoints = 1000000, padding=True, sample_mode='uniform', matlab=False).shuffle()

    dataset.data.x = dataset.data.x.nan_to_num(0.)
    # shuffle dataset and get train/validation/test splits
    num_samples = len(dataset)

    batch_size = 64

    num_val = num_samples // 10

    val_dataset = dataset[:num_val]  # 10
    test_dataset = dataset[num_val:2 * num_val]  # 10
    train_dataset = dataset[2 * num_val:]  # 80

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    #modelpath = 'path to model'
    model = DoubleDeepSetModelBatched(lr=1e-4, input_channels=inp_channels, loss='mape', num_dim=dim)
    #model = model.load_from_checkpoint(modelpath)

    # wandb.init(project='DoubleDeepsetsBatched', name='mape_dim' +str(dim) + '_' + str(inp_channels))
    # wandb_logger = WandbLogger(project='DoubleDeepsetsBatched', name='mape_dim' +str(dim) + '_' + str(inp_channels),
    #                          log_model='all') # log all new checkpoints during training

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    trainer = Trainer(
        accelerator="gpu", devices=1,           # train on GPU
        # logger=wandb_logger,                    # W&B integration
        callbacks=[checkpoint_callback],        # our model checkpoint callback
        max_epochs=200)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()