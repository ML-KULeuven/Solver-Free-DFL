#!/usr/bin/env python
# coding: utf-8
"""
Knapsack problem
"""

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(optGrbModel):
    """
    This class is an optimization model for the knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacities (np.ndarray / listy): Total capacity
        items (list): List of items
    """

    def __init__(self, weights, capacities):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacities (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)
        self.items = self.weights.shape[1]
        self.is_ILP = True
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # Create a model
        m = gp.Model("knapsack")
        # Variables
        x = m.addVars(self.items, name="x", vtype=GRB.BINARY)
        # Sense
        m.modelSense = GRB.MAXIMIZE
        # Constraints
        for i in range(self.weights.shape[0]):
            m.addConstr(gp.quicksum(self.weights[i][j] * x[j] for j in range(self.items)) <= self.capacities[i],
                        name=f"capacity_{i}")
        return m, x

    def relax(self):
        """
        A method to get the linear relaxation model
        """
        # Copy
        model_rel = knapsackModelRel(self.weights, self.capacities)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    This class is a relaxed optimization model for the knapsack problem.
    """

    def _getModel(self):
        """
        Build Gurobi model
        """
        # Create a model
        m = gp.Model("knapsack")
        # Turn off output
        m.Params.outputFlag = 0
        # Variables
        x = m.addVars(self.items, name="x", vtype=GRB.CONTINUOUS, ub=1)
        # Sense
        m.modelSense = GRB.MAXIMIZE
        # Constraints
        for i in range(self.weights.shape[0]):
            m.addConstr(gp.quicksum(self.weights[i][j] * x[j] for j in range(self.items)) <= self.capacities[i],
                        name=f"capacity_{i}")
        return m, x

    def relax(self):
        """
        A forbidden method to relax the relaxed model
        """
        raise RuntimeError("Model has already been relaxed.")

