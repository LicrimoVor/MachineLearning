import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional import structural_similarity_index_measure

from .topoloss_pytorch import getTopoLoss


class Losses:

    __bce_torch = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def bce_loss(logits: torch.Tensor, labels: torch.Tensor):
        losses = logits - logits * labels + torch.log(1 + torch.exp(-logits))
        return losses.mean()

    @classmethod
    def bce_torch(cls, logits: torch.Tensor, labels: torch.Tensor):
        return cls.__bce_torch(logits, labels)

    @staticmethod
    def dice_loss(logits: torch.Tensor, labels: torch.Tensor):
        EPS = 1e-8
        preds = torch.sigmoid(logits)
        inv_labels = (labels == 0).type(torch.uint8)
        TP = (preds * labels).sum()
        FP = (preds * inv_labels).sum()
        FN = (labels - preds * labels).sum()

        score = (2 * TP + EPS) / (2 * TP + FP + FN + EPS)
        return 1 - score

    @classmethod
    def focal_loss(cls, logits: torch.Tensor, labels: torch.Tensor):
        EPS = 1e-8
        GAMMA = 2
        preds = torch.sigmoid(logits) + EPS / 2
        p_t = preds * labels + (1 - preds) * (1 - labels)
        loss = torch.mean((1 - p_t).pow(GAMMA) * cls.bce_loss(logits, labels))

        return loss

    @staticmethod
    def boundary_loss(logits: torch.Tensor, labels: torch.Tensor):
        preds = torch.sigmoid(logits)
        multipled = torch.einsum("bkwh,bkwh->bkwh", preds, labels)
        loss = multipled.sum()
        return loss

    @staticmethod
    def ssim_loss(logits: torch.Tensor, labels: torch.Tensor):
        preds = torch.sigmoid(logits)
        return 1 - structural_similarity_index_measure(preds, labels)

    @staticmethod
    def topo_loss(logits: torch.Tensor, labels: torch.Tensor):
        LAMBDA = 0.1
        topo_ls = getTopoLoss(logits, labels)
        return binary_cross_entropy_with_logits(logits, labels) + LAMBDA * topo_ls

    @staticmethod
    def tversky_loss(logits: torch.Tensor, labels: torch.Tensor):
        ALPHA = 0.3
        BETTA = 0.7
        preds = torch.sigmoid(logits)
        inv_labels = (labels == 0).type(torch.uint8)
        TP = (preds * labels).sum()
        FP = (preds * inv_labels).sum()
        FN = (labels - preds * labels).sum()
        e = 1e-8

        score = (TP + e) / (TP + ALPHA * FP + BETTA * FN + e)
        return 1 - score
