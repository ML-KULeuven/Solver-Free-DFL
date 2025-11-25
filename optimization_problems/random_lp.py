"""
Random linear programming problem
"""
import copy

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog

from pyepo.model.grb.grbmodel import optGrbModel


def is_constraint_implied(A, b, a_k, b_k):
    """
    Check if the constraint a_k x <= b_k is implied by Ax <= b.

    Parameters:
    A (ndarray): (m, n) constraint matrix.
    b (ndarray): (m,) constraint vector.
    a_k (ndarray): (n,) new constraint row vector.
    b_k (float): scalar new constraint bound.

    Returns:
    bool: True if the constraint is implied, False otherwise.
    """

    # Solve LP: maximize a_k @ x subject to Ax <= b
    res = linprog(-a_k, A_ub=A, b_ub=b, method='highs')

    if res.success:
        max_val = -res.fun  # Convert min to max
        return max_val <= b_k  # If max a_k @ x <= b_k, it's redundant
    else:
        raise ValueError("LP solver failed. The constraint system might be infeasible.")


def count_redundant_constraints(A, b):
    """
    A method that counts the number of redundant constraints.
    Args:
        A: The constraint matrix
        b: The right hand side constraint vector

    Returns: the number of redundant constraints.
    """
    A_full = copy.deepcopy(A)
    b_full = copy.deepcopy(b.flatten())

    m = A_full.shape[0]
    redundant_count = 0

    if m == 1:
        return 0

    for i in range(m):
        # Create a new system without the i-th constraint
        A_rest = np.delete(A_full, i, axis=0)
        b_rest = np.delete(b_full, i)

        # Check if A[i] x <= b[i] is implied by the remaining constraints
        if is_constraint_implied(A_rest, b_rest, A_full[i], b_full[i]):
            redundant_count += 1

    return redundant_count


class RandomLP(optGrbModel):

    def __init__(self, num_vars, num_constrs, use_new_generation_method, add_variable_upper_bounds, seed=None):
        self.num_vars = num_vars
        self.num_constrs = num_constrs
        np.random.seed(seed)
        self.is_ILP = False
        super().__init__()

    def _getModel(self):
        """
        A method to build the Gurobi model

        Returns:
            tuple: optimization model and variables
        """

        # Create a model
        m = gp.Model("random_lp")

        # Create variables
        x = m.addVars(self.num_vars, name="x", lb=0)

        # Set model sense
        m.modelSense = GRB.MAXIMIZE

        # Generate random LP
        self.A = np.random.uniform(0, 1, (self.num_constrs, self.num_vars))
        x_star = np.random.uniform(0, 1, self.num_vars)
        b = self.A @ x_star
        additive_term = np.random.uniform(0, 0.2, self.num_constrs)
        random_indices = np.random.choice(self.num_constrs, size=min(self.num_constrs, self.num_vars), replace=False)
        additive_term[random_indices] = 0
        b += additive_term
        self.b = b

        # Ensure there are no redundant constraints
        assert count_redundant_constraints(self.A, self.b) == 0

        # Add constraints
        for i in range(self.num_constrs):
            m.addConstr(
                gp.quicksum(self.A[i, j] * x[j] for j in range(self.num_vars)) <= self.b[i], name=f"Constraint_{i}"
            )

        return m, x