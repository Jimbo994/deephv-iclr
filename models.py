import torch
from torch import nn
from torch.nn import functional as F, MSELoss, L1Loss
from torch.optim import Adam, lr_scheduler
from torchmetrics.functional import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pytorch_lightning import LightningModule

from layers import DoubleDeepSetLayerWithChannelsBatched, DoubleDeepSetLayerWithChannels

class DoubleDeepSetModelBatched(LightningModule):
    def __init__(self, lr=1e-5, input_channels=32, loss='mse', num_dim=3):
        super().__init__()
        self.num_dim = num_dim
        self.lossfn = loss
        self.dds_layer1 = DoubleDeepSetLayerWithChannelsBatched(1, input_channels)
        self.dds_layer2 = DoubleDeepSetLayerWithChannelsBatched(input_channels, input_channels)
        self.dds_layer3 = DoubleDeepSetLayerWithChannelsBatched(input_channels, input_channels)
        self.dds_layer4 = DoubleDeepSetLayerWithChannelsBatched(input_channels, input_channels)
        self.dds_layer5 = DoubleDeepSetLayerWithChannelsBatched(input_channels, 1)

        if self.lossfn == 'mse' or self.lossfn == 'log_mse':
            self.loss = MSELoss()
        if self.lossfn == 'mape':
            self.loss = L1Loss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, batch):
        """ method used for inference input -> output """

        # Now we need to loop over everything in the batch
        # for this we use batch.ptr, we informs us about the shape of each datapoint in the batch

        x = batch.x
        # reshape assuming N and M are constant throughout the batch!!
        x = x.reshape(-1, batch.N[0], batch.M[0])
        #print(x.shape)
        # extract scaling
        initial_scale = x.max(dim=1, keepdim=True)[0]
        initial_scale_inv = 1 / initial_scale
        # in case there was a division by 0
        initial_scale_inv[initial_scale_inv == torch.inf] = 0
        x = x * initial_scale_inv
        shape = x.shape
        x = x.reshape(shape[0], shape[1], shape[2], 1)
        x = self.dds_layer1(x)

        x = F.leaky_relu(x)

        x = self.dds_layer2(x)

        x = F.leaky_relu(x)

        x = self.dds_layer3(x)

        x = F.leaky_relu(x)

        x = self.dds_layer4(x)
        x = F.leaky_relu(x)

        x = self.dds_layer5(x)

        # we can try to add a sigmoid here
        x = F.leaky_relu(x)

        x = x.reshape(shape[0], shape[1], -1)

        denom = 1/(x != 0.).sum(dim=[1,2], keepdim=True)
        denom[denom == torch.inf] = 0
        x = x.sum(dim=[1,2], keepdim=True) * denom

        x = torch.sigmoid(x)

        out = x * initial_scale.prod(dim=2, keepdim=True)

        return out


    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('train_mse', mse)
        self.log('train_mse_log', mse_log)
        self.log('train_map', map)
        self.log('lr', self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        self.log('val_mse_log', mse_log)
        self.log('val_map', map)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        self.log('test_mse_log', mse_log)
        self.log('test_map', map)

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'val_loss'
        }

    def _get_preds_loss_accuracy(self, batch):
        """ convenience function since train/valid/test steps are similar """
        y = batch.y
        preds = self(batch)
        epsilon = 1.17e-06
        if self.lossfn == 'mse':
            loss = self.loss(preds, y.view_as(preds))
        if self.lossfn == 'mape':
            loss = self.loss(preds / y.view_as(preds), y.view_as(preds) / y.view_as(preds))
        if self.lossfn == 'log_mse':
            loss = self.loss(preds.log(), y.view_as(preds).log())

        mae = mean_absolute_error(preds, y.view_as(preds))
        map = mean_absolute_percentage_error(preds, y.view_as(preds))
        mse = mean_squared_error(preds, y.view_as(preds))
        mse_log = mean_squared_error(preds.log(), y.view_as(preds).log())
        return preds, loss, mae, mse, mse_log, map

class DoubleDeepSetModel(LightningModule):
    def __init__(self, lr=1e-5, input_channels=32,  loss='mse', num_dim=3):
        super().__init__()
        self.num_dim = num_dim
        self.lossfn = loss
        self.dds_layer1 = DoubleDeepSetLayerWithChannels(1, input_channels)
        self.dds_layer2 = DoubleDeepSetLayerWithChannels(input_channels, input_channels)
        self.dds_layer3 = DoubleDeepSetLayerWithChannels(input_channels, input_channels)
        self.dds_layer4 = DoubleDeepSetLayerWithChannels(input_channels, input_channels)
        self.dds_layer5 = DoubleDeepSetLayerWithChannels(input_channels, 1)

        if self.lossfn == 'mse' or self.lossfn == 'log_mse':
            self.loss = MSELoss()
        if self.lossfn == 'mape':
            self.loss = L1Loss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, batch):
        '''method used for inference input -> output'''
        # Now we need to loop over everything in the batch
        # for this we use batch.ptr, we informs us about the shape of each datapoint in the batch
        outputs = []
        for i in range(len(batch.ptr)-1):
            x = batch.x[batch.ptr[i]:batch.ptr[i+1]]
            x = x.reshape(batch.N[i], batch.M[i]) #NxM
            # retrieve scaling

            initial_scale = x.abs().max(dim=0)[0]
            initial_scale_inv = 1/initial_scale
            # in case there was a division by 0
            initial_scale_inv[initial_scale_inv==torch.inf] = 0
            x = x * initial_scale_inv

            # reshape to NxMx1
            x = x.reshape(batch.N[i], -1, 1)

            x = self.dds_layer1(x)
            x = F.leaky_relu(x)

            x = self.dds_layer2(x)
            x = F.leaky_relu(x)

            x = self.dds_layer3(x)
            x = F.leaky_relu(x)

            x = self.dds_layer4(x)
            x = F.leaky_relu(x)

            x = self.dds_layer5(x)

            # we can try to add a sigmoid here
            x = F.leaky_relu(x)
            #x = x * scale
            x = x.reshape(batch.N[i], -1)
            x = x.mean(dim=[0,1], keepdim=True)
            x = torch.sigmoid(x)

            x = x * initial_scale.prod()
            outputs.append(x)
        return torch.cat(outputs)

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('train_mse', mse)
        self.log('train_mse_log', mse_log)
        self.log('train_map', map)
        self.log('lr', self.lr)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        self.log('val_mse_log', mse_log)
        self.log('val_map', map)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, mae, mse, mse_log, map = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        self.log('test_mse_log', mse_log)
        self.log('test_map', map)

    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-3)
        #monitor = 'val_loss'
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        y = batch.y
        preds = self(batch)

        if self.lossfn == 'mse':
            loss = self.loss(preds, y.view_as(preds))
        if self.lossfn == 'mape':
            loss = self.loss(preds / y.view_as(preds), y.view_as(preds) / y.view_as(preds))
        if self.lossfn == 'log_mse':
            loss = self.loss(preds.log(), y.view_as(preds).log())

        mae = mean_absolute_error(preds, y.view_as(preds))
        map= mean_absolute_percentage_error(preds, y.view_as(preds))
        mse = mean_squared_error(preds, y.view_as(preds))
        mse_log = mean_squared_error(preds.log(), y.view_as(preds).log())
        return preds, loss, mae, mse, mse_log, map