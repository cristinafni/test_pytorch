# -*- coding: utf-8 -*-
"""losses tailored for 2D-3D pose registration

implements WeightedL2Loss, WeightedL1Loss to account for the fact that out-of-plane is
harder to estimate than in-plane translation and implements GeodesicLoss
as a natural loss for orientation.
"""
import torch
from abc import ABCMeta, abstractmethod
from torch.nn.modules.module import Module
from rotation import (geodesic_loss_matrices, rotation_matrix_from_euler,
                       rotation_matrix_from_ortho6d)
import warnings


class WeightedLoss(Module, metaclass=ABCMeta):
    """Abstract class to create weighted loss functions, set self.loss in child class

    checks that inputs are of equal size and sends them to the device specified by dev

    Note the division by the sum of weights, so just the
    relative weights count - weights=(10, 10, 10) gives the same as = (1,1,1)
    """

    def __init__(self, weight: torch.tensor = None, reduction: str = "mean", dev=None):
        """
        Args:
            weight: None sets every weight to 1, make sure the dtype is compatible with
              the rest of your systems
            reduction: what to do with the results from the different samples
              in the batch. "none", "mean", anything else triggers sum()
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    def forward(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), pred.size()),
                stacklevel=2,
            )
        assert pred.device == target.device, "Please pass predut and target on the same device"
        ret = self.loss(pred, target)

        # if weights are on the wrong device, send them to the right one once
        if self.weight is not None:
            if self.weight.device is not pred.device:
                self.weight = self.weight.to(pred.device)

        if self.weight is None:
            ret = ret.mean(dim=1)
        else:
            ret = torch.matmul(ret, self.weight) / self.weight.sum()
        # reduce results from individual samples in the batch
        if self.reduction != "none":
            ret = torch.mean(ret) if self.reduction == "mean" else torch.sum(ret)
        return ret


class WeightedL2Loss(WeightedLoss):
    r"""calculates \\( (\ \\frac{1}{/sum(w_i)} \sum_i w[i] (pred[i]-target[i])^2) \\) for each sample

    see `deepautomatch.losses.WeightedLoss`
    Examples:
      see `deepautomatch.test_losses`"""

    def __init__(self, weight: torch.tensor = None, reduction: str = "mean", dev=None):
        """see `deepautomatch.losses.WeightedLoss`"""
        super().__init__(weight, reduction, dev)

    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        return (pred - target) ** 2


class WeightedL1Loss(WeightedLoss):
    r"""calculates \\( (\ \\frac{1}{/sum(w_i)} \sum_i w[i] abs(pred[i]-target[i])) \\) for each sample

    see `deepautomatch.losses.WeightedLoss`
    Examples:
      see `deepautomatch.test_losses`"""

    def __init__(self, weight: torch.tensor = None, reduction: str = "mean", dev=None):
        """see `deepautomatch.losses.WeightedLoss`"""
        super().__init__(weight, reduction, dev)

    def loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        return torch.abs(pred - target)


# no testcases yet
# class GeodesicLoss(Module):
#     def __init__(self, reduction:str = "mean"):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, pred:torch.tensor, target:torch.tensor):
#         """see `deepautomatch.rotation.geodesic_loss_matrices`
#         Args:
#             pred, target: shape (batch_nr, 9)
#         """
#         if not (target.size() == pred.size()):
#             warnings.warn(
#                 "Using a target size ({}) that is different to the pred size ({}). "
#                 "This will likely lead to incorrect results due to broadcasting. "
#                 "Please ensure they have the same size.".format(
#                     target.size(), pred.size()
#                 ),
#                 stacklevel=2,
#             )
#         batch_nr = pred.shape[0]
#         pred = pred.view((batch_nr,3,3))
#         targ = target.view((batch_nr, 3,3))
#         return geodesic_loss_matrices(pred, targ)


class GeodesicLossEuler(Module):
    def __init__(self, reduction: str = "mean"):
        """calculates the geodesic loss from Flumatch Euler-Angles Rx, Ry, Rz

        Args:
            reduction: "none", "mean", else: sum()
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.tensor, target: torch.tensor):
        """converts euler to rotation matrices and then calls `deepautomatch.rotation.geodesic_loss_matrices`
        Args:
        pred, target: shape (batch_nr, 3), with typical Flumatch Euler-Angles Rx, Ry, Rz
        """
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), pred.size()),
                stacklevel=2,
            )
        pred_m = rotation_matrix_from_euler(pred[:, 0], pred[:, 1], pred[:, 2])
        target_m = rotation_matrix_from_euler(target[:, 0], target[:, 1], target[:, 2])
        ret = geodesic_loss_matrices(pred_m, target_m)
        if self.reduction != "none":
            ret = torch.mean(ret) if self.reduction == "mean" else torch.sum(ret)
        return ret


class GeodesicLossOrtho6d(Module):
    def __init__(self, reduction: str = "mean"):
        """calculates the geodesic loss from ortho6d rotation represenations

        Args:
        reduction: "none", "mean", else: sum()
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.tensor, target: torch.tensor):
        """converts the ortho6d to rotation matrices and then calls `deepautomatch.rotation.geodesic_loss_matrices`
        Args:
          pred, target: shape (batch_nr, 6), with [0:3] being the first column, [3:6] the second
        """
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), pred.size()),
                stacklevel=2,
            )
        pred_m = rotation_matrix_from_ortho6d(pred)
        target_m = rotation_matrix_from_ortho6d(target)
        ret = geodesic_loss_matrices(pred_m, target_m)
        if self.reduction != "none":
            ret = torch.mean(ret) if self.reduction == "mean" else torch.sum(ret)
        return ret


class CombinedDAMNLoss(Module):
    """Uses transl_loss on  translation components and adds rot_loss on rotation components

    for 2 components:
    elements 0-3 and 6-9 are considered translation components
    elements 3-6 and 9-12 are considered rotation components (Euler angles)

    In the end, we can just call CombinedDAMNLoss()(pred, target) in e.g. damn_core
    """

    def __init__(self, transl_loss, rot_loss, alpha: float = 1.0, components=2):
        """
        Args:
            alpha: tradeoff between the Translation and the RotationLoss
              Loss = TranslLoss(...) + lambda * RotLoss(...)
            components: number of implants

        Example:
            CombinedDAMNLoss(
                WeightedL2Loss([1, 1, 0.1, 1, 1, 0.1]),
                GeodesicLoss())
        """
        super().__init__()
        if components == 1:
            self.comps = [(transl_loss, slice(0, 3, None), 1.0), (rot_loss, slice(3, 6, None), alpha)]
        elif components == 2:
            self.comps = [
                (transl_loss, slice(0, 3, None), 1.0),
                (rot_loss, slice(3, 6, None), alpha),
                (transl_loss, slice(6, 9, None), 1.0),
                (rot_loss, slice(9, 12, None), alpha),
            ]
        else:
            raise NotImplementedError("Invalid number of components")

    def forward(self, pred: torch.tensor, target: torch.tensor):
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), pred.size()),
                stacklevel=2,
            )
        return CombinedLoss(self.comps).forward(pred, target)


class CombinedLoss(Module):
    """takes a list of tuples of the form (loss, slice, weight)"""

    def __init__(self, comps: list):
        super().__init__()
        self.comps = comps

    def forward(self, pred: torch.tensor, target: torch.tensor):
        """tensors have shape (batch_nr, nr_pose_components)"""
        loss = 0
        for (lossfunc, s, w) in self.comps:
            ll = lossfunc(pred[:, s], target[:, s])
            loss += w * ll
        return loss


class JaccardLoss(Module):
    def __init__(self):
        """
        Returns 1 - Jaccard Loss, overall samples in the batch at the same time
        (not the same as taking the loss per image and averaging)
        """
        super().__init__()
        self.eps = 1e-7

    def forward(self, pred: torch.tensor, target: torch.tensor):
        """
        Args:
        pred, target: shape (batch_nr, x, y), typically 1000*1000 images
        """
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), pred.size()),
                stacklevel=2,
            )
        if not ((pred >= 0).all() and (target >= 0).all()):
            warnings.warn("Found non-positive class...")
        target = target.contiguous().view(-1)
        pred = pred.contiguous().view(-1)
        intersection = (pred * target).sum()
        jac = (intersection + self.eps) / (pred.sum() + target.sum() - intersection + self.eps)
        if (jac < 0).any() or (jac > 1).any():
            warnings.warn(
                "Jaccard loss outside allowed range: {}".format(jac),
                stacklevel=2,
            )
        return 1.0 - jac
