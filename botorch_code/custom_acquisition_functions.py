from __future__ import annotations

from typing import Union

import torch
from torch.nn.functional import pad
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def transform(train_obj, ref_point):
    # subtract reference point
    train_obj_shifted = train_obj - ref_point
    # check if no points are worse than ref point
    mask = torch.any(train_obj_shifted < 0, dim=1)
    return train_obj_shifted[~mask]


def transform_mask_non_dominated_allmode(train_obj, ref_point):
    # subtract reference point
    train_obj_shifted = train_obj - ref_point
    shape = train_obj_shifted.shape
    # check if no points are worse than ref point
    mask = torch.any(train_obj_shifted < 0, dim=-1, keepdim=True)

    # pad train_obj_shifted in M dimension
    train_obj_shifted_pad = pad(train_obj_shifted, (0, 10-shape[2], 0, 0), "constant", 1)

    # set all points worse than ref points to zero
    masked = train_obj_shifted * ~mask
    masked_pad = train_obj_shifted_pad * ~mask
    # set all points that are dominated to zero.
    nondom = is_non_dominated(masked).unsqueeze(-1)
    return masked_pad * nondom

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

class UpperConfidenceBoundBaselineDeepHVBatchedPadded(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound on regressed Pareto Front(UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: Model,
            hv_model: Model,
            hv_prev: Tensor,
            train_y: Tensor,
            ref_point: Tensor,
            beta: Union[float, Tensor],
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            B: Projection matrix
            hv_prev: Hypervolume of train_y
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
            y_bound: Bounds on y, so that proposed points are realistic. train_y is normalized so y_bound
                should not be much higher than [1]^d.
        """
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.hv_model = hv_model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.register_buffer("hv_prev", torch.as_tensor(hv_prev))
        self.register_buffer("train_y", torch.as_tensor(train_y))
        self.register_buffer("ref_point", torch.as_tensor(ref_point))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.hv_prev = self.hv_prev.to(X)
        self.train_y = self.train_y.to(X)
        self.ref_point = self.ref_point.to(X)

        posterior = self.model.posterior(X)

        batch_shape = X.shape[:-2]
        mean = posterior.mean
        variance = posterior.variance
        delta = (self.beta.expand_as(mean) * variance).sqrt()

        if self.maximize:
            y = mean + delta

            # duplicate y_train to match batch size
            y_dup = self.train_y.repeat(batch_shape[0], 1, 1)
            # concat proposed designpoints to y_dup
            y_new = torch.cat((y_dup, y.reshape(batch_shape[0], 1, -1)), dim=1)

            # remove points worse than ref points, only keep dominated front, set rest to 0
            # as 0s will be ignored by batched model.

            y_transform = transform_mask_non_dominated_allmode(y_new, self.ref_point)
            shape = y_transform.shape
            y_transform = pad(y_transform, (0,0,0,100-shape[1]), "constant", 0)
            shape = y_transform.shape
            if y_transform.sum(dim=[1,2]).any() == 0:
                hvs = y_transform.sum(dim=[1,2]) * torch.zeros(shape[0])
            else:
                data_list = [Data(x=y_transform[i].flatten().type(torch.float), N=shape[1], M=shape[2],
                     ptr=[0, shape[1] * shape[2]]) for i in range(shape[0])]

                loader = DataLoader(data_list, batch_size=shape[0])
                # should only be one batch this way.

                for batch in loader:
                    hvs = self.hv_model(batch)
            return hvs.flatten() - self.hv_prev

        else:
            raise NotImplementedError(
                " Minimization currently not supported. It is easier to turn your problem into a maximazation "
                "problem instead. "
            )

class UpperConfidenceBoundBaselineDeepHVBatched(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound on regressed Pareto Front(UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: Model,
            hv_model: Model,
            hv_prev: Tensor,
            train_y: Tensor,
            ref_point: Tensor,
            beta: Union[float, Tensor],
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            B: Projection matrix
            hv_prev: Hypervolume of train_y
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
            y_bound: Bounds on y, so that proposed points are realistic. train_y is normalized so y_bound
                should not be much higher than [1]^d.
        """
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.hv_model = hv_model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.register_buffer("hv_prev", torch.as_tensor(hv_prev))
        self.register_buffer("train_y", torch.as_tensor(train_y))
        self.register_buffer("ref_point", torch.as_tensor(ref_point))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.hv_prev = self.hv_prev.to(X)
        self.train_y = self.train_y.to(X)
        self.ref_point = self.ref_point.to(X)

        posterior = self.model.posterior(X)

        batch_shape = X.shape[:-2]
        mean = posterior.mean
        variance = posterior.variance
        delta = (self.beta.expand_as(mean) * variance).sqrt()

        if self.maximize:
            y = mean + delta

            # duplicate y_train to match batch size
            y_dup = self.train_y.repeat(batch_shape[0], 1, 1)
            # concat proposed designpoints to y_dup
            y_new = torch.cat((y_dup, y.reshape(batch_shape[0], 1, -1)), dim=1)

            # remove points worse than ref points, only keep dominated front, set rest to 0
            # as 0s will be ignored by batched model.

            y_transform = transform_mask_non_dominated(y_new, self.ref_point)
            shape = y_transform.shape
            y_transform = pad(y_transform, (0,0,0,100-shape[1]), "constant", 0)
            shape = y_transform.shape
            print(y_transform.sum(dim=[1,2]))
            if y_transform.sum(dim=[1,2]).any() == 0:
                hvs = y_transform.sum(dim=[1,2]) * torch.zeros(shape[0])
            else:
                data_list = [Data(x=y_transform[i].flatten().type(torch.float), N=shape[1], M=shape[2],
                     ptr=[0, shape[1] * shape[2]]) for i in range(shape[0])]

                loader = DataLoader(data_list, batch_size=shape[0])
                # should only be one batch this way.

                for batch in loader:
                    hvs = self.hv_model(batch)
            return hvs.flatten() - self.hv_prev

        else:
            raise NotImplementedError(
                " Minimization currently not supported. It is easier to turn your problem into a maximazation "
                "problem instead. "
            )

class BaselineDeepHVEHVIBatched(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound on regressed Pareto Front(UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: Model,
            hv_model: Model,
            hv_prev: Tensor,
            train_y: Tensor,
            ref_point: Tensor,
            beta: Union[float, Tensor],
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            B: Projection matrix
            hv_prev: Hypervolume of train_y
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
            y_bound: Bounds on y, so that proposed points are realistic. train_y is normalized so y_bound
                should not be much higher than [1]^d.
        """
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.hv_model = hv_model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.register_buffer("hv_prev", torch.as_tensor(hv_prev))
        self.register_buffer("train_y", torch.as_tensor(train_y))
        self.register_buffer("ref_point", torch.as_tensor(ref_point))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.hv_prev = self.hv_prev.to(X)
        self.train_y = self.train_y.to(X)
        self.ref_point = self.ref_point.to(X)

        # This is in R, X is in Z
        #posterior = self._get_posterior(X=X)
        posterior = self.model.posterior(X)

        batch_shape = X.shape[:-2]

        samples = posterior.rsample(sample_shape=torch.Size([32]))
        shape = samples.shape

        if self.maximize:
            y = samples.reshape(-1, 1, shape[-1]) # concat samples and batch shape
            y_dup = self.train_y.repeat(batch_shape[0]*shape[0], 1, 1)
            y_new = torch.cat((y_dup, y.reshape(batch_shape[0]*shape[0], 1, -1)), dim=1)

            # remove points worse than ref points, only keep dominated front, set rest to 0
            # as 0s will be ignored by batched model.
            y_transform = transform_mask_non_dominated(y_new, self.ref_point)

            shape = y_transform.shape
            y_transform = pad(y_transform, (0, 0, 0, 100 - shape[1]), "constant", 0)
            shape = y_transform.shape
            if y_transform.sum(dim=[1, 2]).any() == 0:
                hvs = y_transform.sum(dim=[1, 2]) * torch.zeros(shape[0])
            else:
                data_list = [Data(x=y_transform[i].flatten().type(torch.float), N=shape[1], M=shape[2],
                                  ptr=[0, shape[1] * shape[2]]) for i in range(shape[0])]

                loader = DataLoader(data_list, batch_size=shape[0])
                # should only be one batch this way. NOTE this can become very big if many posterior samples are drawn.
                # scales as MC samples * batch shape.
                for batch in loader:
                    hvs = self.hv_model(batch)
            return hvs.reshape(batch_shape[0], -1).mean(dim=1).nan_to_num() - self.hv_prev

        else:
            raise NotImplementedError(
                " Minimization currently not supported. It is easier to turn your problem into a maximazation "
                "problem instead. "
            )

class BaselineDeepHVEHVIBatchedPadded(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound on regressed Pareto Front(UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: Model,
            hv_model: Model,
            hv_prev: Tensor,
            train_y: Tensor,
            ref_point: Tensor,
            beta: Union[float, Tensor],
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            B: Projection matrix
            hv_prev: Hypervolume of train_y
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
            y_bound: Bounds on y, so that proposed points are realistic. train_y is normalized so y_bound
                should not be much higher than [1]^d.
        """
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.hv_model = hv_model
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.register_buffer("hv_prev", torch.as_tensor(hv_prev))
        self.register_buffer("train_y", torch.as_tensor(train_y))
        self.register_buffer("ref_point", torch.as_tensor(ref_point))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.hv_prev = self.hv_prev.to(X)
        self.train_y = self.train_y.to(X)
        self.ref_point = self.ref_point.to(X)

        # This is in R, X is in Z
        #posterior = self._get_posterior(X=X)
        posterior = self.model.posterior(X)

        batch_shape = X.shape[:-2]

        samples = posterior.rsample(sample_shape=torch.Size([32]))
        shape = samples.shape
        if self.maximize:
            y = samples.reshape(-1, 1, shape[-1]) # concat samples and batch shape
            y_dup = self.train_y.repeat(batch_shape[0]*shape[0], 1, 1)
            y_new = torch.cat((y_dup, y.reshape(batch_shape[0]*shape[0], 1, -1)), dim=1)

            # remove points worse than ref points, only keep dominated front, set rest to 0
            # as 0s will be ignored by batched model.
            y_transform = transform_mask_non_dominated_allmode(y_new, self.ref_point)

            shape = y_transform.shape
            y_transform = pad(y_transform, (0, 0, 0, 100 - shape[1]), "constant", 0)
            shape = y_transform.shape
            if y_transform.sum(dim=[1, 2]).any() == 0:
                print('this is the case')
                hvs = y_transform.sum(dim=[1, 2]) * torch.zeros(shape[0])
            else:
                data_list = [Data(x=y_transform[i].flatten().type(torch.float), N=shape[1], M=shape[2],
                                  ptr=[0, shape[1] * shape[2]]) for i in range(shape[0])]

                loader = DataLoader(data_list, batch_size=shape[0])
                # should only be one batch this way. NOTE this can become very big if many posterior samples are drawn.
                # scales as MC samples * batch shape.
                for batch in loader:
                    hvs = self.hv_model(batch)
            return hvs.reshape(batch_shape[0], -1).mean(dim=1).nan_to_num() - self.hv_prev

        else:
            raise NotImplementedError(
                " Minimization currently not supported. It is easier to turn your problem into a maximazation "
                "problem instead. "
            )


