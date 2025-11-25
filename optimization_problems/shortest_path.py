#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel


class shortestPathModel(optGrbModel):
    """
    This class is an optimization model for the shortest path problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """

    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.arcs = self._getArcs()
        self.is_ILP = False
        super().__init__()

    def _getArcs(self):
        """
        A method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        """
        A method to build the Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # Create a model
        m = gp.Model("shortest path")

        # Variables
        x = m.addVars(self.arcs, name="x")

        # Sense
        m.modelSense = GRB.MINIMIZE

        # Constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # Flow in
                    if v == e[1]:
                        expr += x[e]
                    # Flow out
                    elif v == e[0]:
                        expr -= x[e]
                # Source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # Sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    # m.addConstr(expr == 1)
                    pass
                # Transition
                else:
                    m.addConstr(expr == 0)
        return m, x
