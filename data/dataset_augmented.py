import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from pyepo.model.grb.tsp import tspABModel, tspMTZModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pyepo.model.opt import optModel
from pyepo.data.dataset import optDataset

from utils.misc import count_unique_solutions, find_partial_lexipos, search_for_transition_column, is_lexicofeasible
from utils.slack_model import convert_to_slack_form, getConstraintsMatrixFormSlackModel


class optDatasetAugmented(optDataset):
    def __init__(self, model, feats, costs):
        if not isinstance(model, optModel):
            print("model must be of type optModel")
        self.model = model
        self.feats = feats
        self.costs = costs
        self.adjacent_vertices_cache = {}  # Cache for adjacent vertices
        (self.sols, self.objs, self.relaxed_sols, self.relaxed_objs,
         self.ctrs, self.adjacent_verts, self.adj_vert_computation_time) = self._getSols()

    def _getSols(self):
        num_vars = len(self.model._model.getVars())
        sols, objs, relaxed_sols, relaxed_objs, ctrs, adjacent_verts = [], [], [], [], [], []
        print("Building dataset")
        time.sleep(0.5)
        tbar = tqdm(self.costs)

        # Get LP relaxation of ILP
        if self.model.is_ILP:
            relaxed_model = self.model.relax()

        # Convert to slack form (for computation of adjacent vertices)
        tic = time.time()
        slack_model = convert_to_slack_form(self.model._model)
        slack_model_A, slack_model_b = getConstraintsMatrixFormSlackModel(slack_model)
        self.slack_model_A = slack_model_A
        adj_vert_computation_time = time.time() - tic
        slack_model.optimize()

        # Initialize adjacent vertices cache
        adjacent_vertices_cache = {}

        # Loop through instances
        for i, c in enumerate(tbar):
            # Solve
            sol, obj, model = self._solve(c)

            # Get solution of relaxed model
            if self.model.is_ILP:
                relaxed_model.setObj(c)
                relaxed_sol, relaxed_obj = relaxed_model.solve()
            else:
                relaxed_model = model
                relaxed_sol, relaxed_obj = sol, obj

            # Get binding constraints
            constrs = self._get_binding_constrs(relaxed_model)

            # Find adjacent vertices using slack model
            tic = time.time()
            slack_model_obj = gp.quicksum(c[k] * slack_model.getVars()[k] for k in range(num_vars))
            slack_model.setObjective(slack_model_obj, model._model.ModelSense)
            slack_model.optimize()

            # Create cache key from solution
            cache_key = tuple(sol)
            
            # Check cache for adjacent vertices
            if cache_key in adjacent_vertices_cache:
                adjacent_vertices = adjacent_vertices_cache[cache_key]
            else:
                adjacent_vertices = self._get_adjacent_vertices(slack_model, slack_model_A)
                adjacent_vertices_cache[cache_key] = adjacent_vertices
                adjacent_vertices = [np.array(av) for av in adjacent_vertices]
                for j in range(len(adjacent_vertices)):
                    adjacent_vertices[j] = adjacent_vertices[j][:num_vars]  # Discard slack variables

                # Removing duplicates
                adjacent_vertices = list({tuple(arr.tolist()): arr for arr in adjacent_vertices}.values())

                if len(adjacent_vertices) > 1:
                    sol_array = np.array(sol)
                    adjacent_vertices = [v for v in adjacent_vertices if not np.array_equal(v, sol_array)]
                adjacent_vertices_cache[cache_key] = adjacent_vertices

            adj_vert_computation_time += time.time() - tic

            # Add to data structures
            sols.append(sol)
            objs.append([obj])
            relaxed_sols.append(relaxed_sol)
            relaxed_objs.append([relaxed_obj])
            ctrs.append(np.array(constrs))
            adjacent_verts.append(np.array(adjacent_vertices))

        print(f"Unique solutions: {count_unique_solutions(sols)}")
        print(f"Adjacent vertices computation time: {adj_vert_computation_time}")

        return (np.array(sols),
                np.array(objs),
                np.array(relaxed_sols),
                np.array(relaxed_objs),
                ctrs,
                adjacent_verts,
                adj_vert_computation_time)

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        # Copy model
        model = self.model.copy()
        # Set objective
        model.setObj(cost)
        # Optimize
        sol, obj = model.solve()
        return sol, obj, model

    def _get_binding_constrs(self, model):
        """
        A method to get tight constraints with current solution

        Args:
            model (optModel): optimization models

        Returns:
            np.ndarray: normal vector of constraints
        """
        xs = model._model.getVars()
        constrs = []

        # Iterate over all constraints
        for constr in model._model.getConstrs():
            # Check binding constraints A x == b
            if abs(constr.Slack) < 1e-5:
                t_constr = []
                # Get coefficients
                for x in xs:
                    t_constr.append(model._model.getCoeff(constr, x))

                # Get coefficients with correct direction
                if constr.sense == GRB.LESS_EQUAL:
                    # <=
                    constrs.append(t_constr)
                elif constr.sense == GRB.GREATER_EQUAL:
                    # >=
                    constrs.append([- coef for coef in t_constr])
                elif constr.sense == GRB.EQUAL:
                    # ==
                    constrs.append(t_constr)
                    constrs.append([- coef for coef in t_constr])
                else:
                    # Invalid sense
                    raise ValueError("Invalid constraint sense.")

        # Iterate over all variables to check bounds
        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            # Add tight bounds as constraints
            if x.x <= 1e-5:
                # x_i >= 0
                t_constr[i] = - 1
                constrs.append(t_constr)
            elif x.ub != float('inf') and x.x >= x.ub - 1e-5:
                # x_i <= 1
                t_constr[i] = 1
                constrs.append(t_constr)

        return constrs

    def _get_adjacent_vertices(self, slack_model, A):
        all_vars = slack_model.getVars()
        sol = np.array([var.x for var in all_vars])

        basis_status = np.array([var.VBasis for var in all_vars])  # 0 for basic, -1 or 1 for non-basic
        basic_indices = np.where(basis_status == 0)[0]
        non_basic_indices = np.where(basis_status != 0)[0]

        sigma = np.sum(sol[basic_indices] < 1e-10)

        if sigma == 0:
            # Non-degenerate vertex
            return get_adjacent_vertices_non_degenerate_case(A, basic_indices, non_basic_indices, sol)
        else:
            # Degenerate vertex; use lexicographic pivoting or TNP rule
            use_tnp_rule = False
            return get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_tnp_rule)

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.feats)

    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.relaxed_sols[index]),
            torch.FloatTensor(self.relaxed_objs[index]),
            torch.FloatTensor(self.ctrs[index]),
            torch.FloatTensor(self.adjacent_verts[index])
        )


def collate_fn_2(batch):
    """
    A custom collate function for PyTorch DataLoader.
    """
    # Separate batch data
    x, c, w, z, w_rel, z_rel, t_ctrs, adjacent_verts = zip(*batch)
    # Stack lists of x, c, and w into new batch tensors
    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    z = torch.stack(z, dim=0)
    w_rel = torch.stack(w_rel, dim=0)
    z_rel = torch.stack(z_rel, dim=0)
    # Pad t_ctrs with 0 to make all sequences have the same length
    ctrs_padded = pad_sequence(t_ctrs, batch_first=True, padding_value=0)
    # Pad adjacent_verts with 0 to make all sequences have the same length
    adjacent_verts_tensors = [torch.FloatTensor(av) for av in adjacent_verts]
    adjacent_verts_padded = pad_sequence(adjacent_verts_tensors, batch_first=True, padding_value=0)
    return x, c, w, z, w_rel, z_rel, ctrs_padded, adjacent_verts_padded


def get_adjacent_vertices_non_degenerate_case(A, basic_indices, non_basic_indices, sol):
    A_basic = A[:, basic_indices]
    dirs = np.linalg.solve(-A_basic, A[:, non_basic_indices])
    basic_var_values = sol[basic_indices]

    adjacent_vertices = []
    for i, entering_var_index in enumerate(non_basic_indices):
        dir = dirs[:, i]

        indices_dir_neg = np.where(dir < 0)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = dir[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            complete_dir = np.zeros(A.shape[1])
            complete_dir[basic_indices] = dir
            complete_dir[entering_var_index] = 1
            new_sol = sol + min_ratio * complete_dir
            adjacent_vertices.append(new_sol)

    return adjacent_vertices


def get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_tnp_rule):
    basic_indices = np.array(sorted(basic_indices))
    non_basic_indices = np.array(sorted(non_basic_indices))

    # First determine an initial t (to then construct the t-transition degeneracy graph)
    t_found = False
    while not t_found:
        A_basic = A[:, basic_indices]
        basic_var_values = sol[basic_indices]
        A_non_basic = A[:, non_basic_indices]
        dirs = np.linalg.solve(A_basic, A_non_basic)
        indices_basic_var_zero = np.where(basic_var_values == 0)[0]
        dirs_indices_basic_var_zero = dirs[indices_basic_var_zero, :]
        indices_transition_columns = np.where(np.all(dirs_indices_basic_var_zero <= 0, axis=0))[0]
        if len(indices_transition_columns) > 0:
            t_found = True
        else:
            basic_indices, non_basic_indices = search_for_transition_column(A, basic_indices, non_basic_indices, sol)

    t = non_basic_indices[indices_transition_columns][0]

    B = A[:, basic_indices]
    B_inv = np.linalg.inv(B)
    x_B = sol[basic_indices]
    elements_still_to_fix = np.where(x_B == 0)[0]

    if use_tnp_rule:
        tmp = B_inv @ -A
        cols = find_partial_lexipos(tmp, required_rows=elements_still_to_fix)
        if cols is not None:
            basic_indices_2 = cols
            B_hat = -A[:, basic_indices_2]
            B_hat_prime = np.linalg.solve(B, B_hat)
            B_inv_L = np.concatenate([np.expand_dims(x_B, axis=1), B_hat_prime], axis=1)
            # B_inv_L needs to be lexicofeasible for TNP rule to work
            assert is_lexicofeasible(B_inv_L)
        else:
            B_hat = A[:, basic_indices]
    else:
        B_hat = A[:, basic_indices]

    all_adjacent_vertices = set()
    visited_bases = {tuple(sorted(basic_indices))}

    # Initialize queue with initial basis
    queue = [(basic_indices, non_basic_indices)]
    iteration = 0
    while queue:
        iteration += 1
        ## Early stopping if desired:
        # if iteration > 250:
        #     break
        curr_basic_indices, curr_non_basic_indices = queue.pop(0)

        # Find adjacent vertices for current solution
        adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list = get_adjacent_vertices_degenerate_case_helper(
            A, curr_basic_indices, curr_non_basic_indices, sol, t, B_hat
        )

        for adjacent_vertex in adjacent_vertices:
            all_adjacent_vertices.add(tuple(adjacent_vertex))

        # Add to queue
        for new_basic_indices, new_non_basic_indices in zip(new_basic_indices_list, new_non_basic_indices_list):
            new_basic_indices_tuple = tuple(sorted(new_basic_indices))
            if new_basic_indices_tuple not in visited_bases:
                visited_bases.add(new_basic_indices_tuple)
                queue.append((sorted(new_basic_indices), sorted(new_non_basic_indices)))

    return all_adjacent_vertices


def get_adjacent_vertices_degenerate_case_helper(A, basic_indices, non_basic_indices, sol, t, B_hat):
    A_basic = A[:, basic_indices]
    B_hat_prime = np.linalg.solve(A_basic, B_hat)
    basic_var_values = sol[basic_indices]
    A_non_basic = A[:, non_basic_indices]
    dirs = np.linalg.solve(A_basic, A_non_basic)
    t_index = np.where(non_basic_indices == t)[0]
    x_B = sol[basic_indices]

    # Set numerical stability threshold
    EPSILON = 1e-10

    # Make sure t is a transition column
    indices_basic_var_zero = np.where(basic_var_values == 0)[0]
    transition_column = dirs[indices_basic_var_zero, :]
    transition_column = transition_column[:, t_index]
    assert np.all(transition_column <= EPSILON), "Transition column must have all elements <= 0"

    adjacent_vertices = []
    new_basic_indices_list = []
    new_non_basic_indices_list = []

    for j, entering_var_index in enumerate(non_basic_indices):
            dir = dirs[:, j]
            indices_dir_pos = np.where(dirs[:, j] > EPSILON)[0]
            basic_var_values_dir_pos = basic_var_values[indices_dir_pos]
            positive_dir_values = dir[indices_dir_pos, ]
            ratios = basic_var_values_dir_pos / positive_dir_values

            if ratios.size != 0:
                min_ratio = np.min(ratios)
                min_ratio_indices = indices_dir_pos[np.where(ratios == min_ratio)[0]]

                if min_ratio > 0:
                    # Transition column
                    complete_dir = np.zeros(A.shape[1])
                    complete_dir[basic_indices] = dirs[:, j]
                    complete_dir[entering_var_index] = -1
                    new_sol = sol - min_ratio * complete_dir
                    adjacent_vertices.append(new_sol)

                elif min_ratio == 0:
                    # Find i
                    ratios_2 = dirs[min_ratio_indices, t_index] / dirs[min_ratio_indices, j]
                    max_value = np.max(ratios_2)
                    num_maximizers = np.sum(ratios_2 == max_value)
                    if num_maximizers == 1:
                        i = min_ratio_indices[np.argmax(ratios_2)]
                    else:
                        indices = [q for q in range(len(ratios_2)) if ratios_2[q] == max_value]

                        # Identify indices i where dir[i] > 0
                        candidates = min_ratio_indices[indices]

                        # Construct lexicographic comparison vectors
                        lex_vectors = []
                        for i in candidates:
                            ratio = x_B[i] / dir[i]
                            row_vector = B_hat_prime[i, :] / dir[i]
                            lex_vector = np.concatenate([[ratio], row_vector])
                            lex_vectors.append((i, lex_vector))

                        # Choose i with lexicographically smallest vector
                        i = min(lex_vectors, key=lambda x: tuple(x[1]))[0]

                    # Perform pivot
                    leaving_var_index = basic_indices[i]
                    new_basic_indices = np.copy(basic_indices)
                    new_basic_indices[i] = entering_var_index
                    new_non_basic_indices = np.copy(non_basic_indices)
                    new_non_basic_indices[j] = leaving_var_index

                    # Store results
                    new_basic_indices_list.append(new_basic_indices)
                    new_non_basic_indices_list.append(new_non_basic_indices)

    return adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list
