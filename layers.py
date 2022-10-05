import torch
from torch import nn

class DoubleDeepSetLayer(nn.Module):
    """ Custom permutation equivariant layer
    Uses the mean as a commutative pouling operation.
    """

    def __init__(self):
        super().__init__()
        weights = torch.Tensor(4, 1)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(1, 1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases uniformly
        nn.init.uniform_(self.weights)
        nn.init.uniform_(self.bias)

    def forward(self, x):
        shape = x.shape  # should be NxM or BxNxM

        if len(shape) == 2:
            term1 = self.weights[0] * x
            term2 = (self.weights[1] / shape[0]) * x.sum(dim=0, keepdim=True).repeat(shape[0], 1)
            term3 = (self.weights[2] / shape[1]) * x.sum(dim=1, keepdim=True).repeat(1, shape[1])
            term4 = (self.weights[3] / (shape[0] * shape[1])) \
                    * x.sum(dim=[0, 1], keepdim=True).repeat(shape[0], shape[1])
            term5 = self.bias * torch.ones_like(x)
            out = term1 + term2 + term3 + term4 + term5
            return out
        if len(shape) == 3:
            term1 = self.weights[0] * x
            term2 = (self.weights[1] / shape[1]) * x.sum(dim=1, keepdim=True).repeat(1, shape[1], 1)
            term3 = (self.weights[2] / shape[2]) * x.sum(dim=2, keepdim=True).repeat(1, 1, shape[2])
            term4 = (self.weights[3] / (shape[1] * shape[2])) \
                    * x.sum(dim=[1, 2], keepdim=True).repeat(1, shape[1], shape[2])
            term5 = self.bias * torch.ones_like(x)
            out = term1 + term2 + term3 + term4 + term5
            return out
        else:
            print('Passed tensor has wrong shape, try either batchxNxM or NxM')


class DoubleDeepSetLayerWithChannels(nn.Module):
    """ Custom permutation equivariant layer

    Uses the mean as a commutative pouling operation.
    Supports input and output channels which provides a tuning nob for the expressivity of the model.

    Args:
    input_channels (int): number of input channels of layer
    output_channels (int) number of output channels of layer

    Shape:
        NxMxinput_channels (can handle arbitrary dimensions of N and M, i.e. they can vary over samples.

    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        # weights should have dimensions 5xIinxIout
        weights = torch.Tensor(4, input_channels, output_channels)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(1, output_channels)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.uniform_(self.weights)
        nn.init.uniform_(self.bias)

    def forward(self, x):
        shape = x.shape  # should be NxMxInput size

        # if len(shape) == 2:
        term1 = torch.matmul(x, self.weights[0])  # self.weights[0] * x
        term2 = torch.matmul(x.mean(dim=0, keepdim=True).repeat(shape[0], 1, 1), self.weights[1])
        term3 = torch.matmul(x.mean(dim=1, keepdim=True).repeat(1, shape[1], 1), self.weights[2])
        term4 = torch.matmul(x.mean(dim=[0, 1], keepdim=True).repeat(shape[0], shape[1], 1), self.weights[3])
        term5 = self.bias * torch.ones_like(term4)
        out = term1 + term2 + term3 + term4 + term5
        return out


class DoubleDeepSetLayerWithChannelsBatched(nn.Module):
    """ Custom permutation equivariant layer

    Uses the mean as a commutative pouling operation.
    Supports input and output channels which provides a tuning nob for the expressivity of the model.
    Support batch size > 1.
    Args:
    input_channels (int): number of input channels of layer
    output_channels (int) number of output channels of layer

    Shape:
        BxNxMxinput_channels

    Note:
        In order to utilize batches (B>1), N and M should be fixed for each sample in the batch.
        For this the N dimension needs to be padded with NaN values and the layer will take over correct computation.
    """

    def __init__(self, input_channels, output_channels, do_mask=True):
        super().__init__()
        # weights should have dimensions 5xIinxIout
        weights = torch.Tensor(4, input_channels, output_channels)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(1, output_channels)
        self.bias = nn.Parameter(bias)
        self.do_mask = do_mask

        # initialize weights and biases
        nn.init.kaiming_normal_(self.weights)
        nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        shape = x.shape  # should be BxNxMxInput size

        # Check where values are NaN (or 0)
        mask = x.nan_to_num(nan=0.) != 0

        # Change NaNs to 0s
        #x = x.nan_to_num(nan=0.)

        term1 = torch.matmul(x, self.weights[0])

        denom2 = 1 / (x != 0.).sum(dim=1, keepdim=True)
        denom2[denom2 == torch.inf] = 0
        term2 = x.sum(dim=1, keepdim=True) * denom2

        # repeat to required shape and matmul
        term2 = torch.matmul(term2.repeat(1, shape[1], 1, 1), self.weights[1])

        denom3 = 1/(x != 0.).sum(dim=2, keepdim=True)
        denom3[denom3 == torch.inf] = 0
        term3 = x.sum(dim=2, keepdim=True) * denom3
        term3 = torch.matmul(term3.repeat(1,1,shape[2],1), self.weights[2])

        # take mean, but exclude zeros
        denom4 = 1/(x != 0.).sum(dim=[1, 2], keepdim=True)
        denom4[denom4 == torch.inf] = 0
        term4 = x.sum(dim=[1, 2], keepdim=True) * denom4
        term4 = torch.matmul(term4.repeat(1, shape[1], shape[2], 1), self.weights[3])

        term5 = self.bias * torch.ones_like(term4)

        out = term1 + term2 + term3 + term4 + term5
        return out * mask[:, :, :, 0:1]
