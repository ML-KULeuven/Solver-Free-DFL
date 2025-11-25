import numpy as np
import torch

from pyepo import EPO


def regret(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a neural network for cost prediction
        optmodel (optModel): a PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: the true regret loss
    """

    predmodel.eval()
    loss = 0
    optsum = 0
    # load data
    for data in dataloader:
        if len(data) == 4:
            x, c, w, z = data
        else:
            x, c, w, z, _, _, _, _ = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        with torch.no_grad():  # no grad
            cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calRegret(optmodel, cp[j], c[j].to("cpu").detach().numpy(),
                               z[j].item())
        optsum += abs(z).sum().item()
    # turn back train mode
    predmodel.train()
    # normalized
    return loss / (optsum + 1e-7)


def calRegret(optmodel, pred_cost, true_cost, true_obj):
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_cost (torch.tensor): true costs
        true_obj (torch.tensor): true optimal objective values

    Returns:predmodel
        float: true regret losses
    """
    # Compute optimal solution for predicted cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()

    # Compute objective value with true cost
    if hasattr(optmodel, 'getObj') and callable(optmodel.getObj):
        obj = optmodel.getObj(true_cost, sol)
    else:
        obj = np.dot(sol, true_cost)

    # Loss
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj

    return loss