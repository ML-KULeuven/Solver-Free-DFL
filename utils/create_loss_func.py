import pyepo
import torch
from pyepo.func import NCE, negativeIdentity, perturbedFenchelYoung
from methods.cave import innerConeAlignedCosine, exactConeAlignedCosine
from methods.lava import LAVA


def create_loss_func(loss_type, optmodel, hyperparameters, dataset):

    if loss_type == "mse":
        loss_func = torch.nn.MSELoss()

    elif loss_type == "spo_plus":
        if "solve_ratio" in hyperparameters:
            solve_ratio = hyperparameters["solve_ratio"]
            loss_func = pyepo.func.SPOPlus(optmodel, solve_ratio=solve_ratio, dataset=dataset)
        else:
            loss_func = pyepo.func.SPOPlus(optmodel)

    elif loss_type == "nce":
        solve_ratio = hyperparameters["solve_ratio"]
        loss_func = NCE(optmodel, solve_ratio=solve_ratio, dataset=dataset)

    elif loss_type == "negative_identity":
        if "solve_ratio" in hyperparameters:
            solve_ratio = hyperparameters["solve_ratio"]
            loss_func = negativeIdentity(optmodel, solve_ratio=solve_ratio, dataset=dataset)
        else:
            loss_func = negativeIdentity(optmodel)

    elif loss_type == "pfyl":
        sigma = hyperparameters["sigma"]
        loss_func = perturbedFenchelYoung(optmodel, n_samples=1, sigma=sigma, processes=1)

    elif loss_type == "inner_cave":
        solver = hyperparameters["solver"]
        loss_func = innerConeAlignedCosine(optmodel, solver=solver, processes=1)

    elif loss_type == "exact_cave":
        solver = hyperparameters["solver"]
        loss_func = exactConeAlignedCosine(optmodel, solver=solver, processes=1)

    elif loss_type == "lava":
        loss_func = LAVA(optmodel, threshold=hyperparameters["threshold"])

    else:
        raise Exception("Unknown loss type")

    return loss_func
